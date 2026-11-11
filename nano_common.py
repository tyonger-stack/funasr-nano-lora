"""Fun-ASR-Nano MLX LoRA 微调公共件: prompt 协议 / 音频→speech 嵌入 / 贪心解码。

前向路径逐行对齐 funasr_mlx/llm_asr.py 的 nano 分支(_audio_embeds/_decode),
prompt 采用官方 torch 版格式(model.py generate_chatml):
  system = "You are a helpful assistant."
  user   = "语音转写，不进行文本规整：<|startofspeech|>...<|endofspeech|>"
训练与推理共用同一 build_prompt(), 保证逐字一致。
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import numpy as np

from funasr_mlx.frontend import FbankConfig, WavFrontend, read_audio

IM_END = 151645  # <|im_end|>
EOS = (151643, 151645)

# 官方 ChatML(训练示例与 generate_chatml 同源)
PREFIX_TEXT_OFFICIAL = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n语音转写，不进行文本规整：<|startofspeech|>"
)
TAIL_TEXT_OFFICIAL = "<|endofspeech|><|im_end|>\n<|im_start|>assistant\n"

# funasr_mlx LlmAsrTranscriber 内置协议(仅 sanity 前向等价门对拍用)
PREFIX_TEXT_DEFAULT = (
    "<|im_start|>system\nYou are a speech recognition model.<|im_end|>\n"
    "<|im_start|>user\n语音转写：<|startofspeech|>"
)
TAIL_TEXT_DEFAULT = "<|endofspeech|><|im_end|>\n<|im_start|>assistant\n"

_FBANK_CFG = FbankConfig(
    frame_length=25.0, frame_shift=10.0, dither=0.0,
    window="hamming", n_mels=80, lfr_m=7, lfr_n=6,
)
_frontend = WavFrontend(_FBANK_CFG, cmvn=None)


def load_tokenizer(ckpt_dir: str | Path):
    from transformers import AutoTokenizer

    directory = Path(ckpt_dir)
    for cand in (directory, directory.parent):
        if (cand / "vocab.json").is_file() or (cand / "tokenizer.json").is_file():
            return AutoTokenizer.from_pretrained(str(cand), use_fast=False)
    raise FileNotFoundError(f"No tokenizer files under {directory}(/parent)")


def build_prompt_ids(tok, mode: str = "noitn") -> tuple[list[int], list[int]]:
    """返回 (prefix_ids, tail_ids)。mode: noitn=官方无规整(训练/评测) default=funasr-mlx 内置(对拍用)。"""
    prefix_text = PREFIX_TEXT_OFFICIAL if mode == "noitn" else PREFIX_TEXT_DEFAULT
    tail_text = TAIL_TEXT_OFFICIAL if mode == "noitn" else TAIL_TEXT_DEFAULT
    prefix = tok(prefix_text, add_special_tokens=False)["input_ids"]
    tail = tok(tail_text, add_special_tokens=False)["input_ids"]
    return prefix, tail


def speech_embeds(model, samples: np.ndarray):
    """音频样本 → (adaptor 嵌入 [1,n,d], fake_len)。逐行对齐 llm_asr._audio_embeds nano 分支。"""
    dtype = model.audio_encoder._weight_dtype()
    fb = np.asarray(_frontend(samples), dtype=np.float32)
    t = fb.shape[0]
    olens = 1 + (t - 3 + 2) // 2
    olens = 1 + (olens - 3 + 2) // 2
    fake_len = (olens - 1) // 2 + 1
    speech = mx.array(fb)[None].astype(dtype)
    lengths = mx.array([t], dtype=mx.int32)
    enc, _ = model.audio_encoder(speech, lengths)
    ad = model.audio_adaptor(enc)
    mx.eval(ad)
    return ad, int(fake_len)


def load_audio_for_model(path: str | Path) -> np.ndarray:
    """soundfile 读 flac/wav → 16k mono float32(>30s 由 read_audio 拒绝,与训练过滤一致)。"""
    return read_audio(str(path))


def greedy_decode(model, tok, samples: np.ndarray, *, mode: str = "noitn",
                  max_new_tokens: int = 96) -> list[int]:
    """贪心解码一个样本。镜像 llm_asr._decode nano 分支 + _greedy/_step, 换我们的 prompt。"""
    from funasr_mlx.models.llm import KVCache

    embeds, fake_len = speech_embeds(model, samples)
    n = min(int(embeds.shape[1]), fake_len)
    prefix, tail = build_prompt_ids(tok, mode)
    beg = len(prefix)
    ids = prefix + [0] * fake_len + tail
    emb = model.llm.model.embed_tokens(mx.array([ids], dtype=mx.int32))
    spliced = mx.concatenate(
        [emb[:, :beg, :], embeds[:, :n, :], emb[:, beg + fake_len:, :]], axis=1
    )
    caches = [KVCache() for _ in model.llm.model.layers]
    h = model.llm.model(input_embeddings=spliced, cache=caches, mask="causal")
    logits = model.llm.model.embed_tokens.as_linear(h[:, -1:])
    mx.eval(logits)
    tok_id = int(mx.argmax(logits[0, -1]).item())
    out = [tok_id]
    while len(out) < max_new_tokens and tok_id not in EOS:
        h = model.llm(mx.array([[tok_id]], dtype=mx.int32), cache=caches)
        mx.eval(h)
        tok_id = int(mx.argmax(h[0, -1]).item())
        out.append(tok_id)
    return out


def decode_tokens(tok, token_ids: list[int]) -> str:
    text = tok.decode(token_ids)
    for tag in ("<|im_end|>", "<|endofspeech|>"):
        text = text.replace(tag, "")
    return text.strip()
