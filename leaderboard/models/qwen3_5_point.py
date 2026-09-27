"""ScreenSpot-Pro adapter for Qwen3.5 grounding models trained in deskmind-eyes.

Drop into likaixin2000/ScreenSpot-Pro-GUI-Grounding as `models/qwen3_5_point.py` and register
it in `model_factory.py` (see leaderboard/README.md). This is the protocol our primary metric
uses: a point_2d JSON prompt, native resolution (only images above 16,777,216 pixels are
downscaled), thinking disabled, greedy decoding, 0-1000 relative coordinates.

`PROMPT`, `preprocess_image` and `parse_response` must stay identical to
`deskmind_eyes/common.py` (PROMPT, resize_to_max_pixels(NATIVE_MAX_PIXELS), parse_point);
`tests/test_leaderboard_adapter.py` checks that.

Works for the base model (`Qwen/Qwen3.5-4B`) and for merged LoRA checkpoints exported from
Tinker (`python -m tinker_cookbook.scripts.merge_tinker_adapter_to_hf_model`).
"""

import os
import re

from PIL import Image

MAX_PIXELS = 16_777_216  # Qwen3.5 image processor ceiling (longest_edge)
MAX_NEW_TOKENS = 48

PROMPT = (
    "Locate the UI element in the screenshot that matches the description, and "
    'output its click point in JSON as {{"point_2d": [x, y]}}, with coordinates '
    "in 0-1000 relative to the image.\nDescription: {instruction}"
)

_POINT_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")


def preprocess_image(image: Image.Image) -> Image.Image:
    """Keep native resolution; downscale (LANCZOS) only above MAX_PIXELS."""
    w, h = image.size
    if w * h <= MAX_PIXELS:
        return image
    s = (MAX_PIXELS / (w * h)) ** 0.5
    return image.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)


def build_messages(image: Image.Image, instruction: str) -> list[dict]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT.format(instruction=instruction)},
        ],
    }]


def parse_response(text: str) -> tuple[float, float] | None:
    """First [x, y] after any </think>; 0-1000 ints (or 0-1 floats) -> relative (0-1) point."""
    m = _POINT_RE.search(text.split("</think>")[-1])
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    if x <= 1 and y <= 1:
        return x, y
    return x / 1000, y / 1000


class Qwen3_5PointModel:
    def load_model(self, model_name_or_path: str = "Qwen/Qwen3.5-4B"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path, torch_dtype=dtype,
            attn_implementation=os.environ.get("QWEN3_5_ATTN_IMPL", "sdpa"),
        ).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name_or_path)
        self.generation_kwargs = dict(do_sample=False, max_new_tokens=MAX_NEW_TOKENS)

    def set_generation_config(self, **kwargs):
        # The benchmark script sets temperature=0 / max_new_tokens; greedy decoding is kept.
        if "max_new_tokens" in kwargs:
            self.generation_kwargs["max_new_tokens"] = kwargs["max_new_tokens"]

    def ground_only_positive(self, instruction, image):
        import torch

        if isinstance(image, str):
            assert os.path.isfile(image), "Invalid input image path."
            image = Image.open(image).convert("RGB")
        assert isinstance(image, Image.Image), "Invalid input image."
        image = preprocess_image(image)

        inputs = self.processor.apply_chat_template(
            build_messages(image, instruction), tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, **self.generation_kwargs)
        response = self.processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0]
        point = parse_response(response)
        return {"result": "positive", "format": "x1y1x2y2", "raw_response": response,
                "bbox": None, "point": list(point) if point else None}

    def ground_allow_negative(self, instruction, image):
        raise NotImplementedError()
