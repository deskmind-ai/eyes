"""Stream a diverse subset of ShowUI-web screenshots out of its 43.7GB images.tar.gz.

The tar is grouped by website, so we cap screenshots per site and stop at a target
count or time budget instead of downloading everything. Only screenshots that have
annotations in metadata/hf_train.json are kept; each image is verified with PIL.

python -m deskmind_eyes.prepare_web_subset --target 600 --per-site 15 --max-minutes 25
"""

import argparse
import io
import json
import os
import tarfile
import time
import urllib.request
from collections import Counter

from PIL import Image

from deskmind_eyes.common import DATA_DIR

WEB_DIR = DATA_DIR / "showui_web"
TAR_URL = "https://huggingface.co/datasets/showlab/ShowUI-web/resolve/main/images.tar.gz"
META_URL = "https://huggingface.co/datasets/showlab/ShowUI-web/resolve/main/metadata/hf_train.json"


def rel_path(img_url: str) -> str:
    """'/blob/.../GUI_Exp_Web//images/004_Edge_x/90/screenshot-10.png' -> 'images/004_Edge_x/90/screenshot-10.png'"""
    return "images/" + img_url.split("//images/", 1)[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600)
    ap.add_argument("--per-site", type=int, default=15)
    ap.add_argument("--max-minutes", type=float, default=25)
    args = ap.parse_args()

    WEB_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = WEB_DIR / "hf_train.json"
    if not meta_path.exists():
        urllib.request.urlretrieve(META_URL, meta_path)
    annotated = {rel_path(s["img_url"]) for s in json.load(open(meta_path))}

    have = {str(p.relative_to(WEB_DIR)) for p in (WEB_DIR / "images").rglob("*.png")}
    per_site = Counter(p.split("/")[1] for p in have)
    kept = len(have)
    print(f"{len(annotated)} annotated screenshots; {kept} already on disk", flush=True)

    t0 = time.time()
    resp = urllib.request.urlopen(TAR_URL, timeout=120)
    with tarfile.open(fileobj=resp, mode="r|gz") as tf:
        for m in tf:
            if kept >= args.target or time.time() - t0 > args.max_minutes * 60:
                break
            if not m.isfile() or m.name not in annotated or m.name in have:
                continue
            site = m.name.split("/")[1]
            if per_site[site] >= args.per_site:
                continue
            data = tf.extractfile(m).read()
            try:
                with Image.open(io.BytesIO(data)) as im:
                    im.load()
            except Exception:
                continue
            dst = WEB_DIR / m.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, dst)
            per_site[site] += 1
            kept += 1
            if kept % 50 == 0:
                print(f"{kept} screenshots from {len(per_site)} sites, {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"done: {kept} screenshots from {len(per_site)} sites in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(sorted(per_site.items()))


if __name__ == "__main__":
    main()
