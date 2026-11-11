"""把 LoRA 折叠进基座权重, 产出 funasr-mlx 可直接加载的独立 checkpoint。

W' = W + scale * lora_b @ lora_a   (逐层 q_proj/v_proj, fp32 计算后回原 dtype)

用法:
    python funasr-nano-lora/merge.py --base weights/fun-asr-nano-mlx/fp16 \
        --adapter outputs/lora-nano/henan-aishell/adapter-000600.safetensors \
        --out outputs/merged-nano/henan-aishell
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import numpy as np


from funasr_mlx.pipeline import load_model

from lora import LoRALinear, load_adapter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    base, out = Path(args.base), Path(args.out)
    if out.exists():
        raise SystemExit(f"输出目录已存在: {out}(先删除再合并)")
    out.mkdir(parents=True)

    # 1) 载模型+adapter, 按模块路径统一折叠(路径+".weight" 即 checkpoint 键)
    model, _ = load_model(str(base), family="fun-asr-nano")
    model.freeze()
    load_adapter(args.adapter, model)
    from lora import lora_modules
    mods = lora_modules(model)
    if not mods:
        raise SystemExit("adapter 未含任何 LoRA 模块")
    deltas: dict[str, np.ndarray] = {}
    for p, m in mods.items():
        delta = (m.lora_b.astype(mx.float32) @ m.lora_a.astype(mx.float32)) * m.scale
        deltas[f"{p}.weight"] = np.asarray(delta, dtype=np.float32)
    print(f"[merge] folded {len(deltas)} LoRA tensors: {len(set(p.split('.')[-2] for p in mods))} 种投影")

    # 2) 复制非权重文件(config/tokenizer/index)
    for src in sorted(base.iterdir()):
        if src.name.startswith("model-") and src.suffix == ".safetensors":
            continue
        if src.is_file():
            shutil.copy2(src, out / src.name)

    # 3) 逐分片折叠(用 MLX 原生读写, 兼容 bf16)
    index = json.loads((base / "model.safetensors.index.json").read_text())
    missing = [k for k in deltas if k not in index["weight_map"]]
    if missing:
        raise SystemExit(f"折叠键不在 checkpoint 索引: {missing[:3]}")
    touched = set(index["weight_map"][k] for k in deltas)
    for shard in sorted(base.glob("model-*.safetensors")):
        tensors = dict(mx.load(str(shard)))
        hit = 0
        for key in list(tensors):
            if key in deltas:
                w = tensors[key].astype(mx.float32) + mx.array(deltas[key])
                tensors[key] = w.astype(tensors[key].dtype)
                hit += 1
        # 注意: mx.save_safetensors 只认 .safetensors 后缀, 直接写最终名(失败可整目录重跑)
        mx.save_safetensors(str(out / shard.name), tensors,
                            metadata={"format": "funasr-mlx", "model": "fun-asr-nano"})
        if hit:
            print(f"  {shard.name}: folded {hit}")
    print(f"[done] merged checkpoint → {out}(触碰分片 {len(touched)})")


if __name__ == "__main__":
    main()
