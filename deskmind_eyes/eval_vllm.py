"""ScreenSpot v1 / V2 / Pro evaluation on a GPU with vLLM (cloud eval; no image uploads to Tinker).

Uses the leaderboard adapter's protocol (leaderboard/models/qwen3_5_point.py: PROMPT,
preprocess_image, parse_response, thinking off, greedy, 48 tokens) so scores are comparable with
the Tinker harness (`eval_tinker native=True`) and the official-script adapter.

python -m deskmind_eyes.eval_vllm --model ~/models/Qwen3.5-4B --benchmark screenspot_pro
python -m deskmind_eyes.eval_vllm --model runs/merged/p1-filtered-final --benchmark screenspot_pro --n 500
# merged checkpoints come from: python -m deskmind_eyes.export_ckpt tinker://.../sampler_weights/final
"""

import argparse
import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from deskmind_eyes.common import (BENCHMARKS, MAX_PIXELS_OWL, grounding_messages, load_benchmark, point_in_bbox,
                                  resize_to_max_pixels, summarize)

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("qwen3_5_point", ROOT / "leaderboard" / "models" / "qwen3_5_point.py")
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir or repo id (base or merged checkpoint)")
    ap.add_argument("--benchmark", choices=BENCHMARKS, default="screenspot_pro")
    ap.add_argument("--n", type=int, help="fixed random subset (seed 0), same as eval_tinker n_eval")
    ap.add_argument("--out", help="per-sample jsonl (default runs/eval/vllm_<benchmark>_<model>.jsonl)")
    ap.add_argument("--max-model-len", type=int, default=24_576)  # Pro screenshots reach ~16K image tokens
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--batch", type=int, default=64, help="prompts per vLLM call (bounds decoded images in RAM)")
    ap.add_argument("--max-pixels", type=int,
                    help="downscale to this many pixels like training does (common.MAX_PIXELS = 1003520); "
                         "default keeps native resolution as the leaderboard protocol does")
    ap.add_argument("--zoom", type=float, default=0.0,
                    help="two-pass zoom-in as KV-Ground / GUI-Owl report it: crop this fraction of W and H around the "
                         "first prediction (clipped at the borders), resize the crop to the full image size, predict "
                         "again and map back (0 = off; 0.5 is their setting)")
    ap.add_argument("--first-max-pixels", type=int,
                    help="with --zoom: downscale the first (coarse) pass to this many pixels; the zoom crop still "
                         "uses --max-pixels (native by default). This is the cheap coarse-to-fine setup for a laptop")
    ap.add_argument("--prompt-style", choices=["point", "tool", "owl"], default="point",
                    help="point: our point_2d prompt (training format); tool: Qwen3-VL computer_use tool call")
    args = ap.parse_args()

    import random

    from vllm import LLM, SamplingParams

    samples = load_benchmark(args.benchmark)
    if args.n:
        samples = random.Random(0).sample(samples, args.n)
    tag = Path(args.model.rstrip("/")).name
    suffix = ((f"_mp{args.max_pixels}" if args.max_pixels else "") + (f"_fmp{args.first_max_pixels}" if args.first_max_pixels else "")
              + (f"_zoom{args.zoom}" if args.zoom else "") + (f"_{args.prompt_style}" if args.prompt_style != "point" else "")
              + (f"_n{args.n}" if args.n else ""))
    out = Path(args.out or ROOT / "runs" / "eval" / f"vllm_{args.benchmark}_{tag}{suffix}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=args.model, dtype="bfloat16", max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1}, gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
    )
    params = SamplingParams(temperature=0.0, max_tokens=adapter.MAX_NEW_TOKENS if args.prompt_style == "point" else 128)
    if args.prompt_style == "owl" and not args.max_pixels:  # GUI-Owl's protocol caps images at 9800 visual tokens
        args.max_pixels = MAX_PIXELS_OWL

    def prep(img, max_pixels=None):
        max_pixels = max_pixels or args.max_pixels
        if max_pixels:  # same resize as training (deskmind_eyes.common / data.load_image)
            return resize_to_max_pixels(img, max_pixels)
        return adapter.preprocess_image(img)

    def load(s):
        img = Image.open(s["img_path"]).convert("RGB")
        return prep(img, args.first_max_pixels), img.size

    def predict(batch, images):
        conversations = [grounding_messages(s["instruction"], args.prompt_style, {"type": "image_pil", "image_pil": img})
                         for s, img in zip(batch, images)]
        # "openai" keeps the content parts structured for the HF template; vLLM's auto-detected
        # "string" format joins image placeholder and text with "\n" (one extra prompt token).
        return llm.chat(conversations, params, use_tqdm=False, chat_template_content_format="openai",
                        chat_template_kwargs={"enable_thinking": False})

    def crop_box(point, size):
        """KV-Ground's zoom-in crop (utils.BaseZoomInLazyDataset): centred on the first point, clipped at the borders."""
        W, H = size
        x, y, w, h = point[0] * W, point[1] * H, int(W * args.zoom), int(H * args.zoom)
        return max(0, int(x - w // 2)), max(0, int(y - h // 2)), min(W, int(x + w // 2)), min(H, int(y + h // 2))

    def zoom_image(s, box):
        img = Image.open(s["img_path"]).convert("RGB")
        return prep(img.crop(box).resize(img.size))

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(8) as pool, open(out, "w") as f:
        for i in range(0, len(samples), args.batch):
            batch = samples[i : i + args.batch]
            loaded = list(pool.map(load, batch))
            outputs = predict(batch, [img for img, _ in loaded])
            first = [adapter.parse_response(o.outputs[0].text) for o in outputs]
            second = {}
            if args.zoom:  # second pass on the crops of samples whose first pass parsed
                idx = [j for j, pt in enumerate(first) if pt and 0 <= pt[0] <= 1 and 0 <= pt[1] <= 1]
                boxes = {j: crop_box(first[j], loaded[j][1]) for j in idx}
                crops = list(pool.map(lambda j: zoom_image(batch[j], boxes[j]), idx))
                for j, o in zip(idx, predict([batch[j] for j in idx], crops) if idx else []):
                    pt = adapter.parse_response(o.outputs[0].text)
                    (x1, y1, x2, y2), (W, H) = boxes[j], loaded[j][1]
                    second[j] = (o.outputs[0].text,
                                 ((x1 + pt[0] * (x2 - x1)) / W, (y1 + pt[1] * (y2 - y1)) / H) if pt else first[j])
            for j, (s, (_, size), o) in enumerate(zip(batch, loaded, outputs)):
                text, point = o.outputs[0].text, first[j]
                if j in second:
                    text, point = second[j]
                r = {"img_filename": s["img_filename"], "instruction": s["instruction"],
                     "group": s["group"], "data_type": s["data_type"], "bbox_xyxy": s["bbox_xyxy"],
                     "platform": s.get("platform"),
                     "img_size": list(size), "prompt_tokens": len(o.prompt_token_ids),
                     "output": text, "point": point, "hit": point_in_bbox(s, point, size)}
                if args.zoom:
                    r.update(first_point=first[j], first_hit=point_in_bbox(s, first[j], size))
                rows.append(r)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[{len(rows)}/{len(samples)}] {time.time() - t0:.0f}s, "
                  f"acc {100 * sum(r['hit'] for r in rows) / len(rows):.1f}", flush=True)

    print(json.dumps({**summarize(rows), "benchmark": args.benchmark, "model": args.model,
                      "elapsed_s": round(time.time() - t0), "output_path": str(out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
