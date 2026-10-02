"""Local ScreenSpot eval on Apple Silicon with mlx-vlm (same prompt/parsing as the Tinker eval).

uv tool install mlx-vlm
uvx --from huggingface_hub hf download mlx-community/Qwen3.5-4B-bf16 --local-dir ~/models/Qwen3.5-4B-bf16
~/.local/share/uv/tools/mlx-vlm/bin/python -m deskmind_eyes.eval_mlx ~/models/Qwen3.5-4B-bf16 \
    runs/eval/screenspot_base_mlx.jsonl [--n 30]
Resumable: samples already in the output jsonl are skipped. Also reports speed.
"""

import argparse
import json
import random
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import apply_chat_template, generate, load
from PIL import Image

from deskmind_eyes.common import (
    BENCHMARKS, MAX_PIXELS, NATIVE_MAX_PIXELS, PROMPT, grounding_messages, load_benchmark, parse_point,
    point_in_bbox, resize_to_max_pixels, summarize,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    ap.add_argument("--benchmark", choices=BENCHMARKS, default="screenspot")
    ap.add_argument("--native", action="store_true", help="keep full resolution (slow for Pro)")
    ap.add_argument("--max-pixels", type=int, help="pixel budget per image (the app's grounding server uses 2000000)")
    ap.add_argument("--n", type=int, help="random subset size (seed 0)")
    ap.add_argument("--prompt-style", choices=["point", "tool", "owl"], default="point")
    ap.add_argument("--zoom", type=float, default=0.0,
                    help="two-pass zoom-in: crop this fraction of W/H around the first point and predict again")
    ap.add_argument("--first-max-pixels", type=int,
                    help="with --zoom: pixel budget of the coarse pass (the crop then uses the full budget). "
                         "This is the laptop setup: cheap coarse pass, native-resolution crop")
    ap.add_argument("--max-pixels-crop", type=int, help="pixel budget of the zoom crop (default: --native or 1M)")
    ap.add_argument("--cache-limit-gb", type=float, default=1.0,
                    help="cap MLX's freed-buffer cache (default is ~system RAM)")
    args = ap.parse_args()

    samples = load_benchmark(args.benchmark)
    max_pixels = args.max_pixels or (NATIVE_MAX_PIXELS if args.native else MAX_PIXELS)
    if args.n:
        samples = random.Random(0).sample(samples, args.n)
    key = lambda r: (r["img_filename"], r["instruction"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {key(r): r for r in map(json.loads, open(out))} if out.exists() else {}
    todo = [s for s in samples if key(s) not in done]
    print(f"{len(samples)} samples, {len(done)} done, {len(todo)} to run", flush=True)

    if todo:
        mx.set_cache_limit(int(args.cache_limit_gb * 1024**3))
        model, processor = load(args.model)
        t0 = time.perf_counter()
        with open(out, "a") as f:
            for i, s in enumerate(todo):
                img = Image.open(s["img_path"]).convert("RGB")

                def run(image):
                    """One pass; returns (text, relative point, seconds, prompt tokens)."""
                    if args.prompt_style == "point":
                        chat = PROMPT.format(instruction=s["instruction"])
                    else:  # system prompt with the computer_use tool + the bare instruction, as in eval_vllm
                        system = grounding_messages("x", args.prompt_style)[0]["content"][0]["text"]
                        chat = [{"role": "system", "content": system},
                                {"role": "user", "content": s["instruction"]}]
                    prompt = apply_chat_template(processor, model.config, chat, num_images=1,
                                                 enable_thinking=False)
                    t = time.perf_counter()
                    res = generate(model, processor, prompt, image=[image], max_tokens=64, temperature=0.0)
                    return res, parse_point(res.text), time.perf_counter() - t

                coarse_budget = args.first_max_pixels or max_pixels
                res, point, secs = run(resize_to_max_pixels(img, coarse_budget))
                first_point, zoomed = point, False
                if args.zoom and point and 0 <= point[0] <= 1 and 0 <= point[1] <= 1:
                    W, H = img.size
                    w, h = int(W * args.zoom), int(H * args.zoom)
                    x, y = point[0] * W, point[1] * H
                    box = (max(0, int(x - w // 2)), max(0, int(y - h // 2)),
                           min(W, int(x + w // 2)), min(H, int(y + h // 2)))
                    res2, p2, secs2 = run(resize_to_max_pixels(img.crop(box).resize(img.size),
                                                               args.max_pixels_crop or max_pixels))
                    secs += secs2
                    zoomed = True
                    if p2:
                        point = ((box[0] + p2[0] * (box[2] - box[0])) / W, (box[1] + p2[1] * (box[3] - box[1])) / H)
                    res = res2
                r = {
                    "img_filename": s["img_filename"], "instruction": s["instruction"],
                    "group": s["group"], "data_type": s["data_type"], "bbox_xyxy": s["bbox_xyxy"],
                    "output": res.text, "point": point, "hit": point_in_bbox(s, point, img.size),
                    "first_point": first_point, "first_hit": point_in_bbox(s, first_point, img.size),
                    "zoomed": zoomed,
                    "wall_s": round(secs, 3), "prompt_tokens": res.prompt_tokens,
                    "prefill_tps": round(res.prompt_tps), "decode_tps": round(res.generation_tps, 1),
                    "peak_mem_gb": round(res.peak_memory, 2),
                }
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                done[key(r)] = r
                if (i + 1) % 50 == 0 or i + 1 == len(todo):
                    el = time.perf_counter() - t0
                    gb = lambda b: f"{b / 1024**3:.2f}"
                    print(f"[{i+1}/{len(todo)}] {el/60:.1f}min, eta {el/(i+1)*(len(todo)-i-1)/60:.1f}min | "
                          f"mlx active {gb(mx.get_active_memory())}GB cache {gb(mx.get_cache_memory())}GB "
                          f"peak {gb(mx.get_peak_memory())}GB", flush=True)

    rows = [done[key(s)] for s in samples]
    speed = {k: round(sum(r[k] for r in rows) / len(rows), 1)
             for k in ["wall_s", "prompt_tokens", "prefill_tps", "decode_tps"]}
    print(json.dumps({**summarize(rows), **speed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
