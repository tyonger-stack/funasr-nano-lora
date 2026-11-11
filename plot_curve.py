"""实时训练曲线: loss + lr + 中途 CER 点。用法:
    python funasr-nano-lora/plot_curve.py --run henan-aishell-e1 [--out results/e1_live.png]"""
import argparse, ast, json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
for f in ("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",):
    try: font_manager.fontManager.addfont(f)
    except Exception: pass
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial Unicode MS"

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="henan-aishell-e1")
ap.add_argument("--out", default=None)
ap.add_argument("--max_steps", type=int, default=None, help="横轴上限(默认取日志最大步×1.1)")
a = ap.parse_args()
out = a.out or f"results/{a.run}_live.png"
rows = [ast.literal_eval(l.split("[train] ", 1)[1]) for l in
        (Path("logs") / f"lora_nano_e1.log").read_text().splitlines() if "[train]" in l] \
    if False else [json.loads(l) for l in (Path("outputs/lora-nano") / a.run / "train_log.jsonl").read_text().splitlines()]
X = [r["step"] for r in rows]; Y = [r["loss"] for r in rows]; LR = [r["lr"] for r in rows]
def smooth(xs, ys, w=7):
    ox, oy = [], []
    for i in range(len(xs)):
        lo, hi = max(0, i - w // 2), min(len(xs), i + w // 2 + 1)
        ox.append(xs[i]); oy.append(sum(ys[lo:hi]) / (hi - lo))
    return ox, oy
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
ax.scatter(X, Y, s=10, color="#9ecae1", alpha=.65, label="loss(每50步)")
ax.plot(*smooth(X, Y), color="#1f77b4", lw=2.2, label="loss(7点滑动均值)")
# 中途 CER 点(mid_eval / results)
cers = []
for p in sorted(Path("results").glob(f"cer_{a.run.split('-')[-1]}_step*.json")):
    d = json.loads(p.read_text())
    step = int(p.stem.split("step")[1].split("_")[0])
    cers.append((step, d["cer"] * 10))  # ×10 只为同轴可视化
if cers:
    ax2b = ax.twinx()
    steps, vals = zip(*sorted(cers))
    ax2b.plot(steps, vals, "g^-", ms=9, label="河南CER@300(右轴)")
    ax2b.set_ylabel("河南 CER@300", color="green"); ax2b.tick_params(axis="y", labelcolor="green")
    for j, (s, v) in enumerate(cers): ax2b.annotate(f"{v/10:.3%}", (s, v), textcoords="offset points", xytext=(12, 6 if j % 2 == 0 else -14), fontsize=9, color="green", ha="left")
ax.axvline(600, color="gray", ls=":", lw=1); ax.text(610, max(Y) * .97, "S1-600步=9.28%", fontsize=9, color="#666")
r = rows[-1]
ax.set_title(f"{a.run} 实时曲线 @step {r['step']}  loss {r['loss']:.3f}  lr {r['lr']:.2e}  ({r['elapsed_s']/3600:.1f}h)")
ax.set_ylabel("loss(CE)"); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=.3)
ax2.plot(X, LR, color="#d62728", lw=2); ax2.set_yscale("log")
ax2.set_ylabel("lr"); ax2.set_xlabel("优化步(1 epoch = 19200 步, 每步 32 样本)"); ax2.grid(alpha=.3)
ax.set_xlim(0, a.max_steps or int(max(X) * 1.08))
plt.tight_layout(); plt.savefig(out, dpi=150); print("saved", out)
