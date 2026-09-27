"""GroundCUA functional-instruction records for SFT / RL (87 desktop apps, human-labelled boxes, MIT).

python -m deskmind_eyes.prepare_groundcua --instructions data/groundcua/groundcua_data/functional_instructions.json \
    --out-dir data/groundcua --workers 16

Instructions come from instruction_tuning.tar.gz in ServiceNow/GroundCUA (functional_instructions.json: 167K records over
28.4K screenshots; the paper's ablation found functional instructions best for ScreenSpot-Pro). Screenshots are fetched
one by one from the dataset repo (images/<platform>/<sha256>.png, ~0.37MB each) through HF_ENDPOINT (default huggingface.co;
set HF_ENDPOINT for a mirror). Boxes are pixels in the file; records store them relative, like OS-Atlas.
Output: <out-dir>/records_functional.jsonl, one record per element, with source "groundcua".
"""

import argparse
import json
import os
import random
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
REPO = f"{HF_ENDPOINT}/datasets/ServiceNow/GroundCUA/resolve/main"


def fetch(rel: str, out: Path, attempts: int = 5) -> tuple[int, int] | None:
    """Download images/<rel> to out (atomic); returns the image size, or None if it can't be fetched or decoded."""
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        url = f"{REPO}/images/{urllib.parse.quote(rel)}"
        for i in range(attempts):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "deskmind-eyes"}),
                                            timeout=120) as r:
                    data = r.read()
                tmp = out.with_suffix(".part")
                tmp.write_bytes(data)
                tmp.rename(out)
                break
            except Exception:
                if i == attempts - 1:
                    return None
                time.sleep(2 ** i)
    try:
        with Image.open(out) as im:
            im.load()
            return im.size
    except Exception:
        out.unlink(missing_ok=True)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instructions", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--max-images", type=int, help="random subset of screenshots (seeded)")
    ap.add_argument("--max-per-image", type=int, default=0, help="cap elements per screenshot (0 = keep all)")
    ap.add_argument("--max-area", type=float, default=0.05, help="drop boxes larger than this share of the screen")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    by_img = defaultdict(list)
    for r in json.load(open(args.instructions)):
        if r.get("instruction", "").strip() and r.get("bbox"):
            by_img[f'{r["platform"]}/{r["id"]}.png'].append(r)
    names = sorted(by_img)
    random.Random(args.seed).shuffle(names)
    if args.max_images:
        names = names[: args.max_images]
    print(f"{sum(len(by_img[n]) for n in names)} instructions over {len(names)} screenshots", flush=True)

    img_dir = args.out_dir / "images"
    sizes, t0 = {}, time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        for i, (n, size) in enumerate(zip(names, pool.map(lambda n: fetch(n, img_dir / n), names)), 1):
            if size:
                sizes[n] = size
            if i % 500 == 0 or i == len(names):
                print(f"{i}/{len(names)} screenshots, {len(sizes)} ok, {time.time() - t0:.0f}s", flush=True)

    out = args.out_dir / "records_functional.jsonl"
    kept = dropped = 0
    with open(out, "w") as f:
        for n in names:
            if n not in sizes:
                continue
            W, H = sizes[n]
            els = []
            for r in by_img[n]:
                x1, y1, x2, y2 = r["bbox"]
                b = [max(0.0, x1 / W), max(0.0, y1 / H), min(1.0, x2 / W), min(1.0, y2 / H)]
                if not (b[0] < b[2] and b[1] < b[3]) or (b[2] - b[0]) * (b[3] - b[1]) > args.max_area:
                    dropped += 1
                    continue
                els.append({"source": "groundcua", "img_filename": n, "img_path": str(img_dir / n),
                            "instruction": r["instruction"].strip(), "bbox": b,
                            "point": [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2],
                            "data_type": r.get("category"), "platform": r["platform"], "img_size": [W, H]})
            if args.max_per_image:
                els = els[: args.max_per_image]
            for e in els:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
            kept += len(els)
    print(f"{kept} records -> {out} ({dropped} boxes dropped; {len(names) - len(sizes)} screenshots failed)", flush=True)


if __name__ == "__main__":
    main()
