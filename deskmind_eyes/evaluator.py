"""ScreenSpot (v1 / V2 / Pro) grounding evaluator on Tinker sampling (base model or LoRA checkpoint)."""

import asyncio
import json
import random
from functools import lru_cache
from pathlib import Path

import chz
import tinker
from PIL import Image
from tinker import types

from deskmind_eyes.common import (
    COMPUTER_USE_TOOL, MAX_PIXELS, PROMPT_STYLES, load_benchmark, parse_point_style, point_in_bbox,
    resize_to_max_pixels, summarize,
)
from deskmind_eyes import official_protocol as official
from tinker_cookbook.eval.evaluators import SamplingClientEvaluator
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import ImagePart, Message, TextPart, get_renderer, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer


@lru_cache(maxsize=8)  # Pro screenshots decode to up to ~60MB each
def _load_image(path: Path, max_pixels: int) -> tuple[Image.Image, tuple[int, int]]:
    img = Image.open(path).convert("RGB")
    return resize_to_max_pixels(img, max_pixels), img.size


@lru_cache(maxsize=8)
def _load_image_official(path: Path) -> tuple[Image.Image, tuple[int, int]]:
    """Official ScreenSpot-Pro Qwen3.5 preprocessing: smart_resize(factor=8, max 8294400) + PIL resize."""
    img = Image.open(path).convert("RGB")
    h, w = official.smart_resize(img.height, img.width)
    return img.resize((w, h)), img.size


@chz.chz
class ScreenSpotEvaluatorBuilder:
    model_name: str
    renderer_name: str
    benchmark: str = "screenspot"  # screenspot | screenspot_v2 | screenspot_pro
    max_pixels: int = MAX_PIXELS  # NATIVE_MAX_PIXELS to keep full resolution
    # point | bbox | short | tool (see common.PROMPT_STYLES), or the official ScreenSpot-Pro
    # Qwen3.5 protocol: "official" (HF path: tool system prompt, thinking per renderer) or
    # "official_guided" (vLLM path: assistant turn pre-filled with the tool-call prefix).
    prompt_style: str = "point"
    n_eval: int | None = None  # None = all samples; else a fixed random subset (seed 0)
    max_parallel: int = 64
    max_tokens: int = 48
    output_path: str | None = None  # write per-sample jsonl when set

    def __call__(self) -> "ScreenSpotEvaluator":
        return ScreenSpotEvaluator(self)


class ScreenSpotEvaluator(SamplingClientEvaluator):
    def __init__(self, cfg: ScreenSpotEvaluatorBuilder):
        self.cfg = cfg
        self.renderer = get_renderer(
            cfg.renderer_name, get_tokenizer(cfg.model_name),
            image_processor=get_image_processor(cfg.model_name),
        )
        samples = load_benchmark(cfg.benchmark)
        if cfg.n_eval is not None and cfg.n_eval < len(samples):
            samples = random.Random(0).sample(samples, cfg.n_eval)
        self.samples = samples
        self.tokenizer = get_tokenizer(cfg.model_name)
        self.prefix = (
            self.renderer.create_conversation_prefix_with_tools([COMPUTER_USE_TOOL])
            if cfg.prompt_style == "tool" else []
        )

    async def _run_one(self, s: dict, client: tinker.SamplingClient,
                       params: types.SamplingParams, sem: asyncio.Semaphore) -> dict:
        # Build the prompt only once a slot is free, so at most `max_parallel` encoded
        # images are alive at a time (building all 1272 up front peaked at ~2.9GB RSS).
        if self.cfg.prompt_style.startswith("official"):
            return await self._run_one_official(s, client, params, sem)
        async with sem:
            def build():  # image decode + lossless encode are CPU-heavy: keep them off the event loop
                img, size = _load_image(s["img_path"], self.cfg.max_pixels)
                text_prompt = PROMPT_STYLES[self.cfg.prompt_style].format(instruction=s["instruction"])
                return self.renderer.build_generation_prompt([
                    *self.prefix,
                    Message(role="user", content=[
                        ImagePart(type="image", image=img),
                        TextPart(type="text", text=text_prompt),
                    ]),
                ]), size

            prompt, size = await asyncio.to_thread(build)
            resp = await client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        message = self.renderer.parse_response(resp.sequences[0].tokens)[0]
        text = get_text_content(message)
        if message.get("tool_calls"):
            text += " " + json.dumps([tc.function.arguments if hasattr(tc, "function") else tc
                                      for tc in message["tool_calls"]], default=str)
        point = parse_point_style(text, self.cfg.prompt_style)
        return {
            "img_filename": s["img_filename"], "instruction": s["instruction"],
            "group": s["group"], "data_type": s["data_type"], "bbox_xyxy": s["bbox_xyxy"],
            "img_size": list(size), "prompt_tokens": prompt.length,
            "output": text, "point": point, "hit": point_in_bbox(s, point, size),
        }

    async def _run_one_official(self, s: dict, client: tinker.SamplingClient,
                                params: types.SamplingParams, sem: asyncio.Semaphore) -> dict:
        prefill = official.GUIDED_PREFILL if self.cfg.prompt_style == "official_guided" else None
        async with sem:
            img, size = _load_image_official(s["img_path"])
            prompt = self.renderer.build_generation_prompt([
                Message(role="system", content=official.SYSTEM_TEXT),
                Message(role="user", content=[
                    ImagePart(type="image", image=img),
                    TextPart(type="text", text=s["instruction"]),
                ]),
            ], prefill=prefill)
            resp = await client.sample_async(prompt=prompt, num_samples=1, sampling_params=params)
        tokens = resp.sequences[0].tokens
        text = (prefill or "") + self.tokenizer.decode(tokens, skip_special_tokens=False)
        point = official.parse_response(text)
        return {
            "img_filename": s["img_filename"], "instruction": s["instruction"],
            "group": s["group"], "data_type": s["data_type"], "bbox_xyxy": s["bbox_xyxy"],
            "img_size": list(size), "prompt_tokens": prompt.length, "gen_tokens": len(tokens),
            "output": text, "point": point, "hit": point_in_bbox(s, point, size),
        }

    async def run(self, client: tinker.SamplingClient) -> tuple[dict, list[dict]]:
        params = types.SamplingParams(
            # tool-call answers are longer and may be preceded by a short rationale
            max_tokens=(max(self.cfg.max_tokens, 256) if self.cfg.prompt_style == "tool"
                        else max(self.cfg.max_tokens, 100) if self.cfg.prompt_style.startswith("official")
                        else self.cfg.max_tokens),
            temperature=0.0,
            stop=self.renderer.get_stop_sequences(),
        )
        sem = asyncio.Semaphore(self.cfg.max_parallel)
        rows = await asyncio.gather(*[self._run_one(s, client, params, sem) for s in self.samples])
        if self.cfg.output_path:
            Path(self.cfg.output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cfg.output_path, "w") as f:
                for r in rows:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return summarize(rows), rows

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        table, _ = await self.run(sampling_client)
        return {f"{self.cfg.benchmark}/{k}": float(v) for k, v in table.items()}
