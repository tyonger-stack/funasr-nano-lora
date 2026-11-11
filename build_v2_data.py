"""构建 train_v2.jsonl = asr-mix train(绝对路径) + 郑州伪标注(体检通过的)。"""
import json, sys
from pathlib import Path
import soundfile as sf

out = Path("data/train_v2.jsonl")
n_mix = n_zz = n_drop = 0
with open(out, "w") as fo:
    root = Path("/Users/Apple/datasets/asr-mix/train")
    for line in open("/Users/Apple/datasets/asr-mix/train/metadata.jsonl"):
        r = json.loads(line)
        audio = r["audio"] if r["audio"].startswith("/") else str((root / r["audio"]).resolve())
        fo.write(json.dumps({"audio": audio, "transcription": r["transcription"]}, ensure_ascii=False) + "\n")
        n_mix += 1
    for line in open("data/kespeech_zz_pseudo.jsonl"):
        r = json.loads(line)
        mac = "/Users/Apple/datasets/kespeech-zz/" + r["audio"].lstrip("/")
        try:
            info = sf.info(mac)
            dur = info.frames / info.samplerate
            if info.samplerate != 16000 or not (0.5 <= dur <= 30.0):
                n_drop += 1
                continue
        except Exception:
            n_drop += 1
            continue
        fo.write(json.dumps({"audio": mac, "transcription": r["transcription"]}, ensure_ascii=False) + "\n")
        n_zz += 1
print(f"mix {n_mix} + zz {n_zz}(drop {n_drop}) → {out}")
