"""Send screenshots to Tinker with as little compression loss as the API allows.

tinker_cookbook's `image_to_chunk` re-encodes every image as JPEG with Pillow defaults
(quality 75, 4:2:0 chroma subsampling). UI screenshots have small icons and thin text that
those artifacts destroy: on 100 ScreenSpot-Pro samples the same Qwen3.5-4B scored 51.0 via
Tinker (default JPEG) vs 62.0 locally on MLX (raw pixels), with all 11 disagreements favoring MLX.

The damage comes from 4:2:0 chroma subsampling + q75, not from JPEG itself: JPEG quality 90
with 4:4:4 chroma scored 62.0 on the same 100 samples (lossless 61.0, 95/100 identical, flips
3 vs 2). PNG is ~2x larger (train screenshots: PNG 396 KB, q90-444 155 KB, old default 86 KB),
and big payloads (e.g. 256 RL datums ~ 100 MB) time out on this network.

So the default (GUI_GROUNDING_IMAGE_FORMAT=jpeg90) is JPEG q90 4:4:4, stepping quality down
only to fit Tinker's 2 MiB asset cap. "lossless" = PNG when it fits (else q95..80 4:4:4);
"jpeg" = the cookbook default. Applied on import of deskmind_eyes.common, so evaluation,
difficulty scoring, SFT and RL all see the same pixels.
"""

import io
import os
from collections import Counter

import tinker
from PIL import Image

MAX_ASSET_BYTES = 2_097_152  # Tinker: "Asset is too large ... max allowed is 2097152 bytes"
_LIMIT = MAX_ASSET_BYTES - 16_384  # headroom
DEFAULT_MODE = "jpeg90"
_applied = None
stats: Counter = Counter()  # which encoding was used, for logging


def encode_image(pil_image: Image.Image) -> tuple[bytes, str]:
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    forced = os.environ.get("GUI_GROUNDING_IMAGE_FORMAT", DEFAULT_MODE).lower()
    if forced.startswith("jpeg") and forced[4:].isdigit():  # e.g. jpeg90: JPEG q<=90, 4:4:4
        for quality in range(int(forced[4:]), 49, -5):
            buf = io.BytesIO()
            pil_image.save(buf, format="JPEG", quality=quality, subsampling=0)
            if buf.tell() <= _LIMIT:
                stats[f"jpeg{quality}_444"] += 1
                return buf.getvalue(), "jpeg"
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")  # optimize=True fits no more images and is ~6x slower
    if buf.tell() <= _LIMIT:
        stats["png"] += 1
        return buf.getvalue(), "png"
    for quality in (95, 90, 85, 80):
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=quality, subsampling=0)
        if buf.tell() <= _LIMIT:
            stats[f"jpeg{quality}_444"] += 1
            return buf.getvalue(), "jpeg"
    stats["jpeg75_default"] += 1
    buf = io.BytesIO()
    pil_image.save(buf, format="JPEG", quality=75)
    return buf.getvalue(), "jpeg"


def _image_to_chunk(image_or_str, image_processor):
    from tinker_cookbook.renderers import base

    chunk = base._original_image_to_chunk(image_or_str, image_processor)  # URL loading, token count
    if not isinstance(image_or_str, Image.Image):
        return chunk  # URLs / data URIs: keep the cookbook path
    data, fmt = encode_image(image_or_str)
    return tinker.types.ImageChunk(data=data, format=fmt, expected_tokens=chunk.expected_tokens)


def use_lossless() -> str:
    """Patch every renderer module that imported image_to_chunk. Returns the active mode."""
    global _applied
    mode = os.environ.get("GUI_GROUNDING_IMAGE_FORMAT", DEFAULT_MODE).lower()
    if _applied is not None:
        return _applied
    if mode == "jpeg":  # cookbook default (q75, 4:2:0); "jpeg90" etc. are handled in encode_image
        _applied = "jpeg"
        return _applied
    from tinker_cookbook.renderers import base, qwen3

    if not hasattr(base, "_original_image_to_chunk"):
        base._original_image_to_chunk = base.image_to_chunk
    for module in (base, qwen3):
        module.image_to_chunk = _image_to_chunk
    _applied = mode
    return _applied
