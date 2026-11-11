"""LoRA 注入(任意目标层)、adaptor 投影器 LoRA 与 adapter 保存/加载。

E4 配方(抄服务端 qwen allQKVO+Projector):
    model.freeze()
    apply_lora(model, targets=("q_proj","k_proj","v_proj","o_proj"), r=16, alpha=16)
    apply_lora_adaptor(model, r=16, alpha=16)   # audio_adaptor.linear1/linear2 挂 LoRA(不动块内注意力)

保存/加载按模块路径寻址, merge 按 path+".weight" 对应 checkpoint 键, 全部通用。
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx
import mlx.nn as mxnn


class LoRALinear(mxnn.Module):
    def __init__(self, base: mxnn.Linear, r: int = 16, alpha: float = 16.0,
                 dropout: float = 0.05):
        super().__init__()
        self.base = base  # 已冻结的基座 Linear
        self.r = r
        self.scale = alpha / r
        self.p = float(dropout)
        self.drop_on = False
        out_features, in_features = base.weight.shape
        self.lora_a = mx.random.normal((r, in_features)) * (1.0 / mx.sqrt(r))
        self.lora_b = mx.zeros((out_features, r))

    def __call__(self, x: mx.array) -> mx.array:
        out = self.base(x)
        if self.drop_on and self.p > 0:
            keep = (mx.random.uniform(shape=x.shape) >= self.p).astype(x.dtype)
            x = x * keep / (1.0 - self.p)
        delta = (x @ self.lora_a.T @ self.lora_b.T) * self.scale
        return out + delta.astype(out.dtype)


def _wrap(parent: mxnn.Module, attr: str, r: int, alpha: float, dropout: float) -> bool:
    base = getattr(parent, attr)
    if isinstance(base, LoRALinear):
        return False
    setattr(parent, attr, LoRALinear(base, r, alpha, dropout))
    return True


def apply_lora(model, targets=("q_proj", "v_proj"),
               r: int = 16, alpha: float = 16.0, dropout: float = 0.05) -> int:
    """解码器各层 self_attn 的指定投影挂 LoRALinear(要求先 model.freeze())。"""
    count = 0
    for layer in model.llm.model.layers:
        for proj in targets:
            if _wrap(layer.self_attn, proj, r, alpha, dropout):
                count += 1
    return count


def apply_lora_adaptor(model, r: int = 16, alpha: float = 16.0,
                       dropout: float = 0.0) -> int:
    """audio_adaptor 投影器(linear1/linear2, 对应服务端 multi_modal_projector)挂 LoRA。"""
    count = 0
    for lin in ("linear1", "linear2"):
        if _wrap(model.audio_adaptor, lin, r, alpha, dropout):
            count += 1
    return count


def lora_modules(model) -> dict[str, LoRALinear]:
    """全树收集 LoRALinear: {模块路径: 实例}。路径与 checkpoint 键一一对应。"""
    found: dict[str, LoRALinear] = {}

    def visit(key, mod):
        if isinstance(mod, LoRALinear):
            found[key] = mod

    model.apply_to_modules(visit)
    return found


def lora_params(model) -> dict:
    return {f"{p}.lora_a": m.lora_a for p, m in lora_modules(model).items()} | \
           {f"{p}.lora_b": m.lora_b for p, m in lora_modules(model).items()}


def unfreeze_audio_adaptor(model) -> int:
    """(旧 hybrid 路径, E3 用)整棵解冻 audio_adaptor 升 fp32。"""
    import mlx.utils as mxu
    model.audio_adaptor.unfreeze()
    params = mxu.tree_map(lambda p: p.astype(mx.float32), model.audio_adaptor.parameters())
    model.audio_adaptor.update(params)
    return sum(int(p.size) for _, p in mxu.tree_flatten(model.audio_adaptor.trainable_parameters()))


def count_trainable(model) -> tuple[int, int]:
    def numel(tree) -> int:
        if isinstance(tree, dict):
            return sum(numel(v) for v in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(numel(v) for v in tree)
        return int(tree.size)
    return numel(model.trainable_parameters()), numel(model.parameters())


def save_adapter(path, model):
    """adapter 存 fp32 safetensors + 元数据 sidecar(按模块路径寻址, 通用)。"""
    import json as _json
    mods = lora_modules(model)
    if not mods:
        raise RuntimeError("No LoRA modules to save")
    params = {}
    for p, m in mods.items():
        params[f"{p}.lora_a"] = mx.astype(m.lora_a, mx.float32)
        params[f"{p}.lora_b"] = mx.astype(m.lora_b, mx.float32)
    first = next(iter(mods.values()))
    meta = {"r": first.r, "scale": first.scale, "p": first.p,
            "modules": sorted(mods), "paths_style": "module"}
    mx.save_safetensors(str(path), params)
    Path(str(path) + ".json").write_text(_json.dumps(meta, ensure_ascii=False, indent=1))


def load_adapter(path, model, inject_meta: dict | None = None) -> int:
    """按路径载回。需先按相同配置注入(meta 或 sidecar 记录 modules; decoder targets 自动推导)。"""
    import json as _json
    path = Path(path)
    meta_path = Path(str(path) + ".json")
    meta = _json.loads(meta_path.read_text()) if meta_path.is_file() else (inject_meta or {})
    mods = lora_modules(model)
    if not mods:
        if not meta.get("modules"):
            raise RuntimeError("模型未注入 LoRA 且缺少 sidecar, 无法注入")
        decoder_targets = sorted({m.split(".")[-1] for m in meta["modules"]
                                  if ".self_attn." in m})
        r = int(meta.get("r", 16)); alpha = float(meta.get("scale", 1.0)) * r
        if decoder_targets:
            apply_lora(model, targets=tuple(decoder_targets), r=r, alpha=alpha, dropout=0.0)
        if any(m.startswith("audio_adaptor.") for m in meta["modules"]):
            apply_lora_adaptor(model, r=r, alpha=alpha, dropout=0.0)
        mods = lora_modules(model)
    saved = mx.load(str(path))
    applied = 0
    for p, m in mods.items():
        m.lora_a = mx.astype(saved[f"{p}.lora_a"], m.lora_a.dtype)
        m.lora_b = mx.astype(saved[f"{p}.lora_b"], m.lora_b.dtype)
        applied += 1
    return applied
