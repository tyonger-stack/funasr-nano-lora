"""Fun-ASR-Nano MLX CER 评测。CER 口径逐字复用 qwen3-asr-ft 的 src/cer.py(同数据可比)。

用法:
    # 基座零样本
    python funasr-nano-lora/eval_cer.py --base weights/fun-asr-nano-mlx/fp16 \
        --data_file ~/datasets/asr-mix/eval_henan/metadata.jsonl --tag nano_zeroshot_henan --limit 300
    # LoRA 训后
    python funasr-nano-lora/eval_cer.py --base weights/fun-asr-nano-mlx/fp16 \
        --adapter outputs/lora-nano/henan-aishell/adapter-000600.safetensors \
        --data_file ... --tag nano_lora_henan
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

Q3_REPO = Path("~/.zcode/workspace/798cc683-ac2a-4d54-9a16-213f56d0359c/qwen3-asr-ft").expanduser()
sys.path.insert(0, str(Q3_REPO))
from src.cer import edit_distance, normalize_text_for_cer  # noqa: E402

import mlx.core as mx  # noqa: E402

from funasr_mlx.pipeline import load_model  # noqa: E402

from lora import load_adapter  # noqa: E402
from nano_common import decode_tokens, greedy_decode, load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="funasr-mlx fp16 权重目录")
    ap.add_argument("--adapter", default=None, help="LoRA adapter(.safetensors, 带 sidecar)")
    ap.add_argument("--data_file", required=True, help="metadata.jsonl(audio/transcription)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--prompt_mode", default="noitn", choices=["noitn", "default"])
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    mx.random.seed(args.seed)
    model, _ = load_model(args.base, family="fun-asr-nano")
    if args.adapter:
        model.freeze()
        n = load_adapter(args.adapter, model)
        print(f"[lora] loaded adapter into {n} linears")
    tok = load_tokenizer(args.base)

    items = []
    root = Path(args.data_file).resolve().parent
    with open(args.data_file) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            audio = rec["audio"]
            if not audio.startswith("/"):
                audio = str((root / audio).resolve())
            items.append((audio, rec["transcription"]))
    if args.limit:
        import random
        random.Random(args.seed).shuffle(items)
        items = items[: args.limit]
    print(f"[eval] {len(items)} utterances, tag={args.tag}")

    total_err, total_len, bad = 0, 0, 0
    samples, t0 = [], time.time()
    for idx, (audio, ref) in enumerate(items):
        try:
            import soundfile as sf
            data, _ = sf.read(audio, dtype="float32", always_2d=True)
            data = data.mean(axis=1, dtype="float32")
            ids = greedy_decode(model, tok, data, mode=args.prompt_mode,
                                max_new_tokens=args.max_new_tokens)
            hyp = decode_tokens(tok, ids)
        except Exception as exc:  # 单条失败不拖垮整评
            bad += 1
            hyp = ""
            print(f"[warn] #{idx} {audio}: {exc}", file=sys.stderr)
        ref_n = normalize_text_for_cer(ref)
        hyp_n = normalize_text_for_cer(hyp)
        err = edit_distance(hyp_n, ref_n)
        total_err += err
        total_len += len(ref_n)
        if len(samples) < 400 or err > 0:
            samples.append({"idx": idx, "ref": ref, "hyp": hyp, "err": err, "len": len(ref_n)})
        if (idx + 1) % 50 == 0:
            cer = total_err / max(1, total_len)
            rtf = (time.time() - t0) / (idx + 1)
            print(f"  {idx+1}/{len(items)} CER={cer:.4f} avg={rtf:.2f}s/条", flush=True)

    cer = total_err / max(1, total_len)
    elapsed = time.time() - t0
    result = {"tag": args.tag, "n": len(items), "bad": bad, "cer": round(cer, 4),
              "errors": total_err, "chars": total_len,
              "elapsed_s": round(elapsed, 1), "base": args.base, "adapter": args.adapter,
              "prompt_mode": args.prompt_mode, "limit": args.limit, "samples": samples}
    out = Path(args.out_dir) / f"cer_{args.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"[done] CER={cer:.4f} ({total_err}/{total_len}, bad={bad}) → {out}")


if __name__ == "__main__":
    main()
