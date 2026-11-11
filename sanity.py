"""三道 sanity 门(全过才允许正式训练):

  门1 前向等价: 自写训练态 greedy(用 funasr-mlx 内置 prompt 协议)与官方
       LlmAsrTranscriber 在 N 条上逐 token 一致 → 证明 forward/拼接/解码链路对齐
  门2 单样本过拟合: 1 条样本 ~60 step LoRA 训练, loss → ~0 且转写与标注一致
       → 证明 LoRA 注入/梯度/mask/数据全链路通
  门3 冻结核对: 可训练参数 ≈2.3M(非 8 亿)

用法:
    python funasr-nano-lora/sanity.py --base weights/fun-asr-nano-mlx/fp16 \
        --data_file ~/datasets/asr-mix/eval_henan/metadata.jsonl --n 50
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as mxnn
import mlx.optimizers as mxopt
import numpy as np

from funasr_mlx.llm_asr import LlmAsrTranscriber
from funasr_mlx.pipeline import load_model

from lora import LoRALinear, apply_lora, count_trainable
from nano_common import build_prompt_ids, decode_tokens, greedy_decode, load_tokenizer, speech_embeds
from train import build_train_example, ce_loss, forward_batch, read_metadata

IGNORE = -100


def load_audio(path: str) -> np.ndarray:
    import soundfile as sf
    data, _ = sf.read(path, dtype="float32", always_2d=True)
    return data.mean(axis=1, dtype=np.float32)


def gate1_forward_parity(base: str, data_file: str, n: int) -> bool:
    print("== 门1 前向等价(default 协议, 对拍 LlmAsrTranscriber)==")
    ref_transcriber = LlmAsrTranscriber(base, family="fun-asr-nano")
    model, _ = load_model(base, family="fun-asr-nano")
    tok = load_tokenizer(base)
    items = read_metadata(data_file)[:n]
    mismatch = 0
    for i, it in enumerate(items):
        samples = load_audio(it["audio"])
        ref_ids = ref_transcriber.transcribe_file(it["audio"]).token_ids
        our_ids = greedy_decode(model, tok, samples, mode="default")
        if list(ref_ids) != list(our_ids):
            mismatch += 1
            print(f"  [diff #{i}] ref={ref_ids[:12]}... our={our_ids[:12]}...")
            print(f"    ref_text={ref_transcriber.transcribe_file(it['audio']).text!r}")
    print(f"  门1: {n - mismatch}/{n} 逐 token 一致")
    return mismatch == 0


def gate3_freeze(base: str) -> bool:
    print("== 门3 冻结核对 ==")
    model, _ = load_model(base, family="fun-asr-nano")
    model.freeze()
    n = apply_lora(model)
    n_train, n_total = count_trainable(model)
    print(f"  converted={n} trainable={n_train:,} total={n_total:,}")
    ok = 1_000_000 < n_train < 5_000_000 and n == 56
    return ok


def gate2_overfit(base: str, data_file: str, steps: int = 60, lr: float = 2e-4) -> bool:
    print("== 门2 单样本过拟合 ==")
    model, _ = load_model(base, family="fun-asr-nano")
    model.freeze()
    apply_lora(model)
    for layer in model.llm.model.layers:
        for proj in ("q_proj", "v_proj"):
            getattr(layer.self_attn, proj).drop_on = True
    tok = load_tokenizer(base)
    prefix, tail = build_prompt_ids(tok, "noitn")
    item = read_metadata(data_file)[0]
    print(f"  样本: {item['text']!r} ← {Path(item['audio']).name}")
    spliced, labels = build_train_example(model, tok, prefix, tail, item)
    cur = {"embeds": [spliced], "labels": [labels]}

    def loss_fn():
        return forward_batch(model, cur["embeds"], cur["labels"])

    vg = mxnn.value_and_grad(model, loss_fn)
    opt = mxopt.AdamW(learning_rate=lr)
    first = last = None
    for step in range(steps):
        loss, grads = vg()
        mx.eval(loss, grads)
        if step == 0:
            first = float(loss.item())
        last = float(loss.item())
        opt.update(model, grads)
        if (step + 1) % 10 == 0:
            print(f"  step {step+1}: loss={last:.4f}")
    for layer in model.llm.model.layers:
        for proj in ("q_proj", "v_proj"):
            getattr(layer.self_attn, proj).drop_on = False
    ids = greedy_decode(model, tok, load_audio(item["audio"]), mode="noitn")
    hyp = decode_tokens(tok, ids)
    print(f"  loss {first:.4f} → {last:.4f}; 过拟合转写: {hyp!r}")
    return last < 0.5 and hyp == item["text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--data_file", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--skip", default="", help="逗号分隔要跳过的门, 如 gate2")
    args = ap.parse_args()
    skip = set(args.skip.split(","))
    results = {}
    if "gate3" not in skip:
        results["gate3_freeze"] = gate3_freeze(args.base)
    if "gate1" not in skip:
        results["gate1_forward_parity"] = gate1_forward_parity(args.base, args.data_file, args.n)
    if "gate2" not in skip:
        results["gate2_overfit"] = gate2_overfit(args.base, args.data_file)
    print(json.dumps(results, ensure_ascii=False))
    if not all(results.values()):
        raise SystemExit("SANITY FAILED: " + str(results))
    print("SANITY PASSED")


if __name__ == "__main__":
    main()
