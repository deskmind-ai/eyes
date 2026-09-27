"""Download a desktop subset of OS-Atlas-data (Windows / Linux / macOS) for SFT.

python -m deskmind_eyes.prepare_osatlas --windows 15000 --linux 3000 --macos 2000
python -m deskmind_eyes.prepare_osatlas --windows 150 --linux 30 --macos 20 --out-suffix _smoke

The images live in huge zips (Windows is 71GB split into 4 parts), but their members are
STORED (uncompressed) and the HF CDN honours Range, so we read the zip central directory
over HTTP and pull only the screenshots we sampled. Windows' four parts are presented to
`zipfile` as one virtual file. Records are written to data/osatlas_desktop/records.jsonl
with relative bbox and its center point, matching the ShowUI records in data.py.
"""

import argparse
import http.client
import io
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from deskmind_eyes.common import DATA_DIR

OSATLAS_DIR = DATA_DIR / "osatlas_desktop"
HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
REPO = f"{HF_ENDPOINT}/datasets/OS-Copilot/OS-Atlas-data/resolve/main/desktop_domain"
_HEADERS = {"User-Agent": "deskmind-eyes/0.1 (+prepare_osatlas)"}
# Windows' zip is `split` into four parts that concatenate back into one archive.
PARTS = {"windows": [f"windows_image_a{c}" for c in "abcd"],
         "linux": ["linux_images.zip"], "macos": ["macos_images.zip"]}
ANNOTATIONS = {os_: f"{os_}_splited.json" for os_ in PARTS}


def _request(url: str, headers: dict, attempts: int = 6) -> bytes:
    """GET with retries: the overseas CDN drops connections mid-body (IncompleteRead) under parallel load."""
    for i in range(attempts):
        try:
            req = urllib.request.Request(url, headers={**_HEADERS, **headers})
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except (http.client.IncompleteRead, urllib.error.URLError, ConnectionError, TimeoutError) as e:
            if i == attempts - 1:
                raise
            time.sleep(2 ** i)
    raise AssertionError("unreachable")


class HttpZipFile(io.RawIOBase):
    """Seekable read-only file over one or more HTTP objects concatenated in order."""

    def __init__(self, urls: list[str]):
        self.urls = urls
        self.sizes = [self._size(u) for u in urls]
        self.total = sum(self.sizes)
        self.pos = 0

    @staticmethod
    def _size(url: str) -> int:
        req = urllib.request.Request(url, headers=_HEADERS, method="HEAD")
        with urllib.request.urlopen(req, timeout=60) as r:
            # HF redirects to a CDN; x-linked-size is the real object size when present.
            return int(r.headers.get("x-linked-size") or r.headers["Content-Length"])

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.total}[whence]
        self.pos = max(0, min(self.total, base + offset))
        return self.pos

    def readinto(self, b) -> int:
        n = min(len(b), self.total - self.pos)
        if n <= 0:
            return 0
        b[:n] = self.read_range(self.pos, n)
        self.pos += n
        return n

    def read_range(self, offset: int, length: int) -> bytes:
        """Read `length` bytes at a global offset, spanning parts when needed."""
        out = bytearray()
        for url, size in zip(self.urls, self.sizes):
            if length <= 0:
                break
            if offset >= size:
                offset -= size
                continue
            take = min(length, size - offset)
            out += _request(url, {"Range": f"bytes={offset}-{offset + take - 1}"})
            offset, length = 0, length - take
        return bytes(out)


def sample_records(os_: str, n: int, max_per_image: int, max_area: float, seed: int,
                   full_only: bool = False) -> list[dict]:
    """Pick whole screenshots (shuffled) until `n` valid elements are collected."""
    path = OSATLAS_DIR / ANNOTATIONS[os_]
    if not path.exists():  # the Windows annotations are a single 268MB JSON array
        print(f"{os_}: downloading {ANNOTATIONS[os_]}")
        path.write_bytes(_request(f"{REPO}/{ANNOTATIONS[os_]}", {}))
    entries = json.load(open(path))
    by_image: dict[str, list] = {}
    for e in entries:  # the same screenshot appears in several entries
        by_image.setdefault(e["img_filename"], []).extend(e.get("elements", []))

    names = sorted(by_image)
    if full_only:  # "*_sub<k>.png" are 1280x720-ish crops; Pro screenshots are 2.5-8x larger
        names = [n for n in names if not re.search(r"_sub\d+", n)]
    random.Random(seed).shuffle(names)
    records, kept = [], 0
    for name in names:
        if kept >= n:
            break
        elements = []
        for el in by_image[name]:
            x1, y1, x2, y2 = el["bbox"]
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                continue  # ~0.3% of boxes fall outside the image
            if not el.get("instruction", "").strip():
                continue
            if (x2 - x1) * (y2 - y1) > max_area:
                continue  # a11y containers (~0.4%); Pro targets are far smaller (median 0.035% of the screen)
            elements.append({
                "source": f"osatlas_{os_}", "img_filename": name,
                "img_path": str(OSATLAS_DIR / "images" / os_ / name),
                "instruction": el["instruction"].strip(),
                "bbox": [x1, y1, x2, y2], "point": [(x1 + x2) / 2, (y1 + y2) / 2],
                "data_type": el.get("data_type"),
            })
        if not elements:
            continue
        elements = elements[:max_per_image]
        records += elements
        kept += len(elements)
    return records


def fetch_images(os_: str, names: set[str], workers: int = 8) -> set[str]:
    """Extract the given members from the remote zip; returns the names that failed."""
    import zipfile

    out_dir = OSATLAS_DIR / "images" / os_
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = sorted(n for n in names if not (out_dir / n).exists())
    if not todo:
        return set()
    raw = HttpZipFile([f"{REPO}/{p}" for p in PARTS[os_]])
    print(f"{os_}: zip {raw.total / 2**30:.1f} GiB, reading central directory")
    with zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20)) as zf:
        # Members are stored under a directory prefix and duplicated under several names.
        by_base: dict[str, zipfile.ZipInfo] = {}
        for info in zf.infolist():
            if not info.is_dir():
                by_base.setdefault(Path(info.filename).name, info)
        def fetch(name: str) -> bool:
            info = by_base.get(name)
            if info is None or info.compress_type != zipfile.ZIP_STORED:
                return False
            header = raw.read_range(info.header_offset, 30)
            name_len = int.from_bytes(header[26:28], "little")
            extra_len = int.from_bytes(header[28:30], "little")
            data = raw.read_range(info.header_offset + 30 + name_len + extra_len, info.file_size)
            tmp = out_dir / f".{name}.part"
            tmp.write_bytes(data)
            try:
                with Image.open(tmp) as im:
                    im.load()
            except Exception:
                tmp.unlink(missing_ok=True)
                return False
            tmp.rename(out_dir / name)
            return True

        # Each member is two small HTTP range requests to an overseas CDN: latency-bound, so fan out.
        failed = set()
        with ThreadPoolExecutor(workers) as pool:
            for i, (name, ok) in enumerate(zip(todo, pool.map(fetch, todo)), 1):
                if not ok:
                    failed.add(name)
                if i % 50 == 0 or i == len(todo):
                    print(f"{os_}: {i}/{len(todo)} images", flush=True)
    return failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", type=int, default=15000)
    ap.add_argument("--linux", type=int, default=3000)
    ap.add_argument("--macos", type=int, default=2000)
    ap.add_argument("--max-per-image", type=int, default=25, help="cap elements per screenshot")
    ap.add_argument("--max-area", type=float, default=0.05, help="drop boxes larger than this share of the screen")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--full-only", action="store_true", help="only uncropped full-resolution screenshots")
    args = ap.parse_args()
    OSATLAS_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    for os_ in ("windows", "linux", "macos"):
        n = getattr(args, os_)
        if not n:
            continue
        records = sample_records(os_, n, args.max_per_image, args.max_area, args.seed, args.full_only)
        failed = fetch_images(os_, {r["img_filename"] for r in records})
        records = [r for r in records if r["img_filename"] not in failed]
        print(f"{os_}: {len(records)} records over "
              f"{len({r['img_filename'] for r in records})} screenshots ({len(failed)} images failed)")
        all_records += records

    out = OSATLAS_DIR / f"records{args.out_suffix}.jsonl"
    with open(out, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(all_records)} records -> {out}")


if __name__ == "__main__":
    main()
