"""Find GroundCUA screenshots that duplicate UI-Vision ones (same 83 apps, same authors) and drop them from training.

python -m deskmind_eyes.dedup_uivision --groundcua data/groundcua --ui-vision data/ui_vision --max-dist 6

Perceptual difference hash (dHash, 16x16 = 256 bits) of every screenshot; a pair within `max-dist` bits is a
(near-)duplicate. Writes <groundcua>/uivision_duplicates.json (pairs + distances) and
<groundcua>/records_functional_dedup.jsonl without the matched screenshots.
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

HASH = 16


def dhash(path: str) -> bytes:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((HASH + 1, HASH), Image.Resampling.LANCZOS), dtype=np.int16)
    return np.packbits((g[:, 1:] > g[:, :-1]).ravel()).tobytes()


def hashes(paths: list[str], workers: int) -> np.ndarray:
    with ProcessPoolExecutor(workers) as pool:
        hs = list(pool.map(dhash, paths, chunksize=64))
    return np.frombuffer(b"".join(hs), dtype=np.uint8).reshape(len(paths), -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groundcua", type=Path, default=Path("data/groundcua"))
    ap.add_argument("--ui-vision", type=Path, default=Path("data/ui_vision"))
    ap.add_argument("--max-dist", type=int, default=6, help="Hamming distance (of 256 bits) counted as a duplicate")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    gc = sorted(str(p) for p in (args.groundcua / "images").rglob("*.png"))
    uv = sorted(str(p) for p in (args.ui_vision / "images").rglob("*.png"))
    print(f"hashing {len(gc)} GroundCUA and {len(uv)} UI-Vision screenshots", flush=True)
    hg, hu = hashes(gc, args.workers), hashes(uv, args.workers)
    bits = np.unpackbits(hg, axis=1).astype(np.uint8)
    pairs = []
    for i, u in enumerate(np.unpackbits(hu, axis=1).astype(np.uint8)):
        d = (bits != u).sum(1)
        for j in np.nonzero(d <= args.max_dist)[0]:
            pairs.append({"ui_vision": uv[i], "groundcua": gc[j], "dist": int(d[j])})
    dup = {Path(p["groundcua"]).relative_to(args.groundcua / "images").as_posix() for p in pairs}
    print(f"{len(pairs)} pairs within {args.max_dist} bits; {len({p['ui_vision'] for p in pairs})} UI-Vision and "
          f"{len(dup)} GroundCUA screenshots involved", flush=True)
    (args.groundcua / "uivision_duplicates.json").write_text(json.dumps(pairs, indent=1))

    kept = dropped = 0
    with open(args.groundcua / "records_functional_dedup.jsonl", "w") as f:
        for line in open(args.groundcua / "records_functional.jsonl"):
            if json.loads(line)["img_filename"] in dup:
                dropped += 1
                continue
            f.write(line)
            kept += 1
    print(f"records_functional_dedup.jsonl: {kept} kept, {dropped} dropped", flush=True)


if __name__ == "__main__":
    main()
