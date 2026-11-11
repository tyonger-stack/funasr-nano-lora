"""Fun-ASR-Nano MLX LoRA 训练器。直接读 asr-mix 的 metadata.jsonl({"audio","transcription"})。

用法:
    python funasr-nano-lora/train.py --config funasr-nano-lora/configs/henan-aishell-nano.yaml
    python funasr-nano-lora/train.py --config ... --resume outputs/lora-nano/henan-aishell/ckpt-000600

批构造: 逐样本算 speech 嵌入(encoder/adaptor 单条前向), 拼接成全序列嵌入后 pad 组批,
LLM 一次前向(cache=None 自动 causal), CE 只对 assistant 段。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as mxnn
import mlx.optimizers as mxopt
import numpy as np
import yaml
from mlx.utils import tree_flatten, tree_map

from funasr_mlx.pipeline import load_model

from lora import LoRALinear, apply_lora, apply_lora_adaptor, count_trainable, load_adapter, save_adapter
from nano_common import IM_END, build_prompt_ids, load_tokenizer, speech_embeds

IGNORE = -100


def read_metadata(path: str | Path) -> list[dict]:
    root = Path(path).resolve().parent
    items = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audio = rec["audio"]
            if not audio.startswith("/"):
                audio = str((root / audio).resolve())
            items.append({"audio": audio, "text": rec["transcription"]})
    return items


def audio_seconds(path: str) -> float:
    import soundfile as sf
    return sf.info(path).frames / sf.info(path).samplerate


class Batcher:
    """shuffle → 局部按时长排序(省 pad)→ 微批。"""

    def __init__(self, items, batch_size: int, sort_pool: int, seed: int):
        self.items, self.bsz, self.pool, self.rng = items, batch_size, sort_pool, random.Random(seed)

    def __iter__(self):
        order = list(range(len(self.items)))
        self.rng.shuffle(order)
        for base in range(0, len(order), self.pool):
            chunk = sorted(order[base:base + self.pool],
                           key=lambda i: self.dur[i])
            for s in range(0, len(chunk), self.bsz):
                yield [self.items[i] for i in chunk[s:s + self.bsz]]

    def attach_durations(self, cache_path: str | None = None):
        import json as _json
        cache = {}
        if cache_path and Path(cache_path).is_file():
            cache = _json.loads(Path(cache_path).read_text())
        durs = []
        missing = []
        for it in self.items:
            if it["audio"] in cache:
                durs.append(cache[it["audio"]])
            else:
                missing.append(it["audio"])
        for p in missing:
            cache[p] = audio_seconds(p)
        Path(cache_path).write_text(_json.dumps(cache))
        self.dur = [cache[it["audio"]] for it in self.items]


def build_train_example(model, tok, prefix, tail, item):
    """单样本 → (spliced 嵌入 [1,L,d], labels_full [L])。"""
    import soundfile as sf
    samples, _ = sf.read(item["audio"], dtype="float32", always_2d=True)
    samples = samples.mean(axis=1, dtype=np.float32)
    embeds, fake_len = speech_embeds(model, samples)
    n = min(int(embeds.shape[1]), fake_len)
    target = tok(item["text"], add_special_tokens=False)["input_ids"] + [IM_END]
    ids = prefix + [0] * fake_len + tail + target
    emb = model.llm.model.embed_tokens(mx.array([ids], dtype=mx.int32))
    beg = len(prefix)
    spliced = mx.concatenate(
        [emb[:, :beg, :], embeds[:, :n, :], emb[:, beg + fake_len:, :]], axis=1
    )
    L = int(spliced.shape[1])
    labels = [IGNORE] * L
    first_pred = beg + n + len(tail) - 1  # 该位置的输出预测 target[0]
    for j, tid in enumerate(target):
        pos = first_pred + j
        if pos < L:
            labels[pos] = tid
    return spliced, labels


def ce_loss(logits, labels):
    """logits (B,L,V), labels (B,L) 含 IGNORE。"""
    logits = logits.astype(mx.float32)
    logp = logits - mx.log(mx.sum(mx.exp(logits), axis=-1, keepdims=True))
    _, _, V = logp.shape
    flat = logp.reshape(-1, V)
    tgt = labels.reshape(-1)
    valid = tgt != IGNORE
    idx = mx.arange(flat.shape[0])
    picked = flat[idx, mx.maximum(tgt, 0)]
    return -(picked * valid).sum() / valid.sum()


def forward_batch(model, batch_embeds, batch_labels):
    L = max(e.shape[1] for e in batch_embeds)
    d = batch_embeds[0].shape[2]
    dtype = batch_embeds[0].dtype
    embs = mx.zeros((len(batch_embeds), L, d), dtype=dtype)
    labels = np.full((len(batch_embeds), L), IGNORE, dtype=np.int64)
    for i, (e, lab) in enumerate(zip(batch_embeds, batch_labels)):
        embs[i, : e.shape[1]] = e[0]
        labels[i, : len(lab)] = lab
    labels_m = mx.array(labels)
    h = model.llm.model(input_embeddings=embs)
    logits = model.llm.model.embed_tokens.as_linear(h)
    return ce_loss(logits, labels_m)


def build_lr_schedule(lr, scheduler, max_steps, warmup_steps):
    warmup_steps = max(1, int(warmup_steps))
    if scheduler == "constant":
        return lambda step: lr
    if scheduler == "cosine":
        def fn(step):
            if step < warmup_steps:
                return lr * (step + 1) / warmup_steps
            prog = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return lr * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))
        return fn
    raise ValueError(scheduler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", default=None, help="adapter 文件路径")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    name = cfg["name"]
    out_dir = Path(cfg.get("output_dir", "outputs/lora-nano")) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    lc = cfg.get("lora_config", {})
    tc = cfg.get("training_args", {})
    max_steps = int(tc.get("max_steps", 600))
    bsz = int(tc.get("per_device_train_batch_size", 2))
    accum = int(tc.get("gradient_accumulation_steps", 16))
    grad_clip = float(tc.get("max_grad_norm", 1.0))
    grad_skip = float(tc.get("grad_norm_skip", 100.0))
    log_steps = int(tc.get("logging_steps", 10))
    save_steps = int(tc.get("save_steps", 100))
    keep = int(tc.get("save_total_limit", 2))
    seed = int(tc.get("seed", 1883))

    mx.random.seed(seed)
    random.seed(seed)

    model, _ = load_model(cfg["model_config"]["base_dir"], family="fun-asr-nano")
    model.freeze()
    n_conv = apply_lora(model,
                        targets=tuple(lc.get("target_modules", ["q_proj", "v_proj"])),
                        r=int(lc.get("r", 16)),
                        alpha=float(lc.get("lora_alpha", 32)),
                        dropout=float(lc.get("dropout", 0.05)))
    n_ad = 0
    if lc.get("adaptor_projector", False):
        n_ad = apply_lora_adaptor(model,
                                  r=int(lc.get("r", 16)),
                                  alpha=float(lc.get("lora_alpha", 32)),
                                  dropout=float(lc.get("dropout", 0.0)))
    hc = cfg.get("hybrid_config", {})
    if hc.get("unfreeze_audio_adaptor", False):
        from lora import unfreeze_audio_adaptor
        unfreeze_audio_adaptor(model)
        print("[hybrid] audio_adaptor 整棵解冻(旧 E3 路径)")
    if args.resume:
        load_adapter(args.resume, model)
    n_train, n_total = count_trainable(model)
    print(f"[lora] decoder {n_conv} + adaptor {n_ad} linears; trainable {n_train:,} / total {n_total:,}")
    if n_train > 40_000_000 or (n_conv + n_ad) == 0:
        raise SystemExit("冻结核对失败: 可训练参数异常, 停止")

    tok = load_tokenizer(cfg["model_config"]["base_dir"])
    prefix, tail = build_prompt_ids(tok, "noitn")
    print(f"[prompt] prefix {len(prefix)} ids / tail {len(tail)} ids")

    items = read_metadata(cfg["data_config"]["data_file"])
    print(f"[data] {len(items)} utterances from {cfg['data_config']['data_file']}")
    batcher = Batcher(items, bsz, sort_pool=bsz * accum * 8, seed=seed)
    dur_cache = str(out_dir / "dur_cache.json")
    batcher.attach_durations(dur_cache)

    warmup_cfg = float(tc.get("warmup_steps", 0.1))
    warmup_n = int(warmup_cfg * max_steps) if warmup_cfg < 1 else int(warmup_cfg)  # <1 为比例, ≥1 为绝对步数
    lr_fn = build_lr_schedule(float(tc.get("learning_rate", 1e-4)),
                              tc.get("lr_scheduler_type", "cosine"),
                              max_steps, warmup_n)
    print(f"[sched] warmup {warmup_n} steps (config={warmup_cfg}), peak lr {tc.get('learning_rate', 1e-4)}")
    opt = mxopt.AdamW(learning_rate=lr_fn(0), weight_decay=float(tc.get("weight_decay", 0.01)))

    def loss_fn():
        return forward_batch(model, cur["embeds"], cur["labels"])

    vg = mxnn.value_and_grad(model, loss_fn)

    opt_step = 0
    accum_count = 0
    accum_grads = None
    skipped_steps = 0
    t0 = time.time()
    log_f = open(out_dir / "train_log.jsonl", "a")
    ckpts: list[Path] = []
    running = []

    # LoRA dropout 仅在训练路径生效(全树所有 LoRALinear)
    from lora import lora_modules
    lora_mods = lora_modules(model)
    for mod in lora_mods.values():
        mod.drop_on = True

    def maybe_save(force=False):
        nonlocal ckpts
        if force or (opt_step % save_steps == 0):
            path = out_dir / f"adapter-{opt_step:06d}.safetensors"
            save_adapter(path, model)
            ckpts.append(path)
            while len(ckpts) > keep:
                ckpts.pop(0).unlink(missing_ok=True)
            print(f"[ckpt] saved {path}", flush=True)

    epoch = 0
    stop = False
    while not stop:
        epoch += 1
        for micro in batcher:
            batch_embeds, batch_labels = [], []
            for it in micro:
                e, lab = build_train_example(model, tok, prefix, tail, it)
                batch_embeds.append(e)
                batch_labels.append(lab)
            cur = {"embeds": batch_embeds, "labels": batch_labels}
            loss, grads = vg()
            mx.eval(loss, grads)
            loss_v = float(loss.item())
            gnorm_v = float(mx.sqrt(sum((g * g).sum() for _, g in tree_flatten(grads))).item())
            running.append(loss_v)
            if not math.isfinite(gnorm_v) or gnorm_v > grad_skip:
                skipped_steps += 1
                print(f"[fuse] grad_norm={gnorm_v:.1f} skip (total {skipped_steps})", flush=True)
            else:
                gscale = min(1.0, grad_clip / gnorm_v) if grad_clip > 0 else 1.0
                grads = tree_map(lambda g: g * gscale, grads)
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
                accum_count += 1
            if accum_count >= accum:
                accum_grads = tree_map(lambda g: g / accum, accum_grads)
                opt.learning_rate = lr_fn(opt_step)
                opt.update(model, accum_grads)
                mx.eval(model.parameters())
                accum_grads, accum_count = None, 0
                opt_step += 1
                maybe_save()
                if opt_step % log_steps == 0:
                    avg = sum(running[-log_steps * accum:]) / max(1, len(running[-log_steps * accum:]))
                    msg = {"step": opt_step, "loss": round(avg, 4), "lr": lr_fn(opt_step),
                           "grad_skip": skipped_steps, "elapsed_s": round(time.time() - t0, 1)}
                    print(f"[train] {msg}", flush=True)
                    log_f.write(json.dumps(msg) + "\n")
                    log_f.flush()
                if opt_step >= max_steps:
                    stop = True
                    break
        if stop:
            break
    maybe_save(force=True)
    print(f"[done] opt_steps={opt_step} skipped={skipped_steps} elapsed={(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
