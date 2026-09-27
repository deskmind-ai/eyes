"""Grounding training records (ShowUI-desktop, ShowUI-web subset) and the SFT dataset.

Each Datum = one screenshot + one description -> {"point_2d": [x, y]}.
We deliberately avoid ShowUI's multi-turn packing (many queries per screenshot):
the Qwen3.5 renderer drops the empty <think></think> block from earlier assistant
turns, so multi-turn training would not match the single-turn inference prompt.
"""

import json
import math
import random
from functools import lru_cache

import chz
import tinker
from PIL import Image

from deskmind_eyes.common import (
    DATA_DIR, MAX_PIXELS, PROMPT, SHOWUI_DESKTOP_DIR, format_answer, resize_to_max_pixels,
)

SHOWUI_WEB_DIR = DATA_DIR / "showui_web"
TRAIN_SOURCES = ["showui_desktop", "showui_web", "osatlas_desktop", "osatlas_full_v3"]
OSATLAS_DIR = DATA_DIR / "osatlas_desktop"
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import ImagePart, Message, TextPart, TrainOnWhat, get_renderer
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder
from tinker_cookbook.tokenizer_utils import get_tokenizer


def load_showui_desktop(holdout_images: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Flatten to one record per (screenshot, instruction); split by screenshot."""
    screenshots = json.load(open(SHOWUI_DESKTOP_DIR / "hf_train.json"))
    random.Random(seed).shuffle(screenshots)
    splits = []
    for part in (screenshots[holdout_images:], screenshots[:holdout_images]):
        splits.append([
            {"source": "showui_desktop", "img_url": s["img_url"],
             "img_path": str(SHOWUI_DESKTOP_DIR / "images" / s["img_url"]),
             "instruction": e["instruction"], "point": e["point"], "bbox": e["bbox"]}
            for s in part
            for e in s["element"]
        ])
    return splits[0], splits[1]


def load_showui_web() -> list[dict]:
    """Records for the ShowUI-web screenshots present on disk (see prepare_web_subset.py)."""
    records = []
    for s in json.load(open(SHOWUI_WEB_DIR / "hf_train.json")):
        rel = "images/" + s["img_url"].split("//images/", 1)[1]
        path = SHOWUI_WEB_DIR / rel
        if not path.exists():
            continue
        for e in s["element"]:
            records.append({"source": "showui_web", "img_url": rel, "img_path": str(path),
                            "instruction": e["instruction"], "point": e["point"], "bbox": e["bbox"]})
    return records


def load_osatlas(holdout_images: int = 0, seed: int = 0, name: str = "records.jsonl") -> tuple[list[dict], list[dict]]:
    """OS-Atlas desktop records (see prepare_osatlas.py); split by screenshot like ShowUI.
    name="records_full_v3.jsonl": full-resolution screenshots with Pro-style rewritten instructions
    (rewritten by a frontier model, see docs/training.md); same 30-screenshot holdout as train_hf.py."""
    records = [json.loads(l) for l in open(OSATLAS_DIR / name)]
    images = sorted({r["img_path"] for r in records})
    random.Random(seed).shuffle(images)
    held = set(images[:holdout_images])
    return ([r for r in records if r["img_path"] not in held],
            [r for r in records if r["img_path"] in held])


def load_train_records(sources: str, seed: int = 0) -> list[dict]:
    """Comma-separated sources -> shuffled records with a stable `key` (source|image|instruction).
    ShowUI-desktop uses the same 6-screenshot holdout as SFT."""
    records = []
    for src in (s for s in sources.split(",") if s):
        if src == "showui_desktop":
            records += load_showui_desktop(holdout_images=6, seed=0)[0]
        elif src == "showui_web":
            records += load_showui_web()
        elif src == "osatlas_desktop":
            records += load_osatlas(holdout_images=30, seed=0)[0]
        elif src == "osatlas_full_v3":
            records += load_osatlas(holdout_images=30, seed=0, name="records_full_v3.jsonl")[0]
        else:
            raise ValueError(f"unknown source {src!r}; choose from {TRAIN_SOURCES}")
    for r in records:
        r["key"] = f"{r['source']}|{r.get('img_url') or r['img_filename']}|{r['instruction']}"
    records = list({r["key"]: r for r in records}.values())
    random.Random(seed).shuffle(records)
    return records


# Native-resolution screenshots are up to ~7MP (21MB decoded), so keep the cache small.
@lru_cache(maxsize=32)
def load_image(img_path: str, max_pixels: int = MAX_PIXELS) -> Image.Image:
    return resize_to_max_pixels(Image.open(img_path).convert("RGB"), max_pixels)


class GroundingSFTDataset(SupervisedDataset):
    def __init__(self, records: list[dict], model_name: str, renderer_name: str,
                 batch_size: int, max_length: int, seed: int = 0):
        self.records = records
        self.batch_size = batch_size
        self.max_length = max_length
        self.renderer = get_renderer(
            renderer_name, get_tokenizer(model_name), image_processor=get_image_processor(model_name)
        )
        self.set_epoch(seed)

    def set_epoch(self, seed: int = 0):
        self.order = list(range(len(self.records)))
        random.Random(seed).shuffle(self.order)

    def __len__(self) -> int:
        return math.ceil(len(self.records) / self.batch_size)

    def build_datum(self, rec: dict) -> tinker.Datum:
        messages = [
            Message(role="user", content=[
                ImagePart(type="image", image=load_image(rec["img_path"])),
                TextPart(type="text", text=PROMPT.format(instruction=rec["instruction"])),
            ]),
            Message(role="assistant", content=format_answer(*rec["point"])),
        ]
        model_input, weights = self.renderer.build_supervised_example(
            messages, train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )
        return datum_from_model_input_weights(model_input, weights, max_length=self.max_length)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        idxs = self.order[index * self.batch_size : (index + 1) * self.batch_size]
        return [self.build_datum(self.records[i]) for i in idxs]


@chz.chz
class RecordsBuilder(SupervisedDatasetBuilder):
    """SFT over any comma-separated TRAIN_SOURCES mix, holding out whole screenshots for NLL."""

    model_name: str
    renderer_name: str
    sources: str = "osatlas_desktop"
    batch_size: int = 32
    max_length: int = 4096
    holdout_images: int = 30
    split_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train, held = [], []
        for src in (x for x in self.sources.split(",") if x):
            if src == "osatlas_desktop":
                t, h = load_osatlas(self.holdout_images, self.split_seed)
            elif src == "showui_desktop":
                t, h = load_showui_desktop(self.holdout_images, self.split_seed)
            elif src == "showui_web":
                t, h = load_showui_web(), []
            else:
                raise ValueError(f"unknown source {src!r}; choose from {TRAIN_SOURCES}")
            train += t
            held += h
        kw = dict(model_name=self.model_name, renderer_name=self.renderer_name,
                  batch_size=self.batch_size, max_length=self.max_length)
        return GroundingSFTDataset(train, **kw), (GroundingSFTDataset(held, **kw) if held else None)


@chz.chz
class ShowUIDesktopBuilder(SupervisedDatasetBuilder):
    model_name: str
    renderer_name: str
    batch_size: int = 32
    max_length: int = 4096
    holdout_images: int = 6
    split_seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        train, held = load_showui_desktop(self.holdout_images, self.split_seed)
        kw = dict(model_name=self.model_name, renderer_name=self.renderer_name,
                  batch_size=self.batch_size, max_length=self.max_length)
        return GroundingSFTDataset(train, **kw), (GroundingSFTDataset(held, **kw) if held else None)
