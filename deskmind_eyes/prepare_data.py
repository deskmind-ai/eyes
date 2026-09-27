"""Download ScreenSpot v1 / V2 / Pro (eval) and ShowUI-desktop (SFT/RL train) into data/.

python -m deskmind_eyes.prepare_data            # download missing / broken files
python -m deskmind_eyes.prepare_data --verify   # only check, no download

Every image is written to a temp file, size-checked against the HF `x-linked-size`
header, decoded with PIL, then atomically renamed. Re-runs skip files that decode.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from deskmind_eyes.common import (
    SCREENSPOT_DIR, SCREENSPOT_PRO_DIR, SCREENSPOT_V2_DIR, SHOWUI_DESKTOP_DIR,
)

HF_ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")  # e.g. https://hf-mirror.com
HF = f"{HF_ENDPOINT}/datasets"
# hf-mirror.com answers 403 to urllib's default "Python-urllib/x.y" User-Agent.
_HEADERS = {"User-Agent": "deskmind-eyes/0.1 (+prepare_data)"}


def _open(u: str, timeout: float = 60):
    return urllib.request.urlopen(urllib.request.Request(u, headers=_HEADERS), timeout=timeout)


def _pro_annotations() -> dict[str, str]:
    files = json.load(_open(f"{HF_ENDPOINT}/api/datasets/likaixin/ScreenSpot-Pro"))
    names = [f["rfilename"] for f in files["siblings"] if f["rfilename"].startswith("annotations/")]
    return {n: n for n in names}


SOURCES = {
    "screenspot": dict(
        repo="KevinQHLin/ScreenSpot", dir=SCREENSPOT_DIR,
        metadata={f"{d}.json": f"metadata/screenspot_{d}.json" for d in ["mobile", "desktop", "web"]},
        images=lambda d: sorted({s["img_filename"] for f in ["mobile", "desktop", "web"]
                                 for s in json.load(open(d / f"{f}.json"))}),
    ),
    "screenspot_v2": dict(
        repo="OS-Copilot/ScreenSpot-v2", dir=SCREENSPOT_V2_DIR,
        metadata={f"screenspot_{d}_v2.json": f"screenspot_{d}_v2.json" for d in ["mobile", "desktop", "web"]},
        images=lambda d: sorted({s["img_filename"] for f in ["mobile", "desktop", "web"]
                                 for s in json.load(open(d / f"screenspot_{f}_v2.json"))}),
        # Images are only published as one 1.3GB zip with a top-level folder.
        archive=("screenspotv2_image.zip", "screenspotv2_image/"),
    ),
    "screenspot_pro": dict(
        repo="likaixin/ScreenSpot-Pro", dir=SCREENSPOT_PRO_DIR,
        metadata=_pro_annotations,
        images=lambda d: sorted({s["img_filename"] for f in (d / "annotations").glob("*.json")
                                 for s in json.load(open(f))}),
    ),
    "showui_desktop": dict(
        repo="showlab/ShowUI-desktop", dir=SHOWUI_DESKTOP_DIR,
        metadata={"hf_train.json": "metadata/hf_train.json"},
        images=lambda d: sorted({s["img_url"] for s in json.load(open(d / "hf_train.json"))}),
    ),
}


def url(repo: str, path: str) -> str:
    return f"{HF}/{repo}/resolve/main/{urllib.parse.quote(path)}"


def image_ok(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.load()
        return True
    except Exception:
        return False


def fetch(src: str, dst: Path, check_image: bool, retries: int = 3) -> str | None:
    """Download src -> dst atomically. Returns an error string, or None on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    err = "unknown"
    for _ in range(retries):
        try:
            with _open(src) as resp, open(tmp, "wb") as f:
                expected = resp.headers.get("Content-Length")
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
            size = tmp.stat().st_size
            if expected is not None and size != int(expected):
                err = f"size {size} != expected {expected}"
                continue
            if check_image and not image_ok(tmp):
                err = "PIL cannot decode"
                continue
            os.replace(tmp, dst)
            return None
        except Exception as e:  # network errors: retry
            err = repr(e)
    tmp.unlink(missing_ok=True)
    return err


def extract_from_archive(cfg: dict, root: Path, todo: list[str]) -> list[tuple[str, str]]:
    """Download the dataset zip (size-checked, atomic), extract only the needed images, verify each."""
    zip_name, prefix = cfg["archive"]
    zip_path = root / zip_name
    if not zip_path.exists():
        print(f"downloading {zip_name} ...")
        if err := fetch(url(cfg["repo"], zip_name), zip_path, check_image=False):
            return [(n, f"archive download failed: {err}") for n in todo]
    failed = []
    with zipfile.ZipFile(zip_path) as z:
        members = set(z.namelist())
        for n in todo:
            member = prefix + n
            if member not in members:
                failed.append((n, "not in archive"))
                continue
            dst = root / "images" / n
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + ".part")
            with z.open(member) as src, open(tmp, "wb") as f:
                while chunk := src.read(1 << 20):
                    f.write(chunk)
            if image_ok(tmp):
                os.replace(tmp, dst)
            else:
                tmp.unlink(missing_ok=True)
                failed.append((n, "PIL cannot decode"))
    if not failed:
        zip_path.unlink()  # images extracted and verified; the 1.3GB zip is no longer needed
    return failed


def prepare(name: str, verify_only: bool, workers: int) -> bool:
    cfg = SOURCES[name]
    root: Path = cfg["dir"]
    metadata = cfg["metadata"]() if callable(cfg["metadata"]) else cfg["metadata"]
    for local, remote in metadata.items():
        if not (root / local).exists():
            if verify_only:
                print(f"[{name}] missing metadata {local}")
                return False
            if err := fetch(url(cfg["repo"], remote), root / local, check_image=False):
                print(f"[{name}] metadata {local} failed: {err}")
                return False

    images = cfg["images"](root)
    with ThreadPoolExecutor(workers) as pool:
        ok = list(pool.map(lambda n: image_ok(root / "images" / n), images))
    todo = [n for n, good in zip(images, ok) if not good]
    print(f"[{name}] {len(images)} images, {len(images) - len(todo)} ok, {len(todo)} missing/broken")
    if verify_only or not todo:
        return not todo

    if "archive" in cfg:
        failed = extract_from_archive(cfg, root, todo)
    else:
        with ThreadPoolExecutor(workers) as pool:
            errors = list(pool.map(
                lambda n: fetch(url(cfg["repo"], f"images/{n}"), root / "images" / n, check_image=True),
                todo,
            ))
        failed = [(n, e) for n, e in zip(todo, errors) if e]
    for n, e in failed[:10]:
        print(f"[{name}] FAILED {n}: {e}")
    print(f"[{name}] downloaded {len(todo) - len(failed)}, failed {len(failed)}")
    return not failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=list(SOURCES))
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    names = [args.only] if args.only else list(SOURCES)
    results = [prepare(n, args.verify, args.workers) for n in names]
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
