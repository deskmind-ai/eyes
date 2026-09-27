"""Score RL records by the policy's own pass rate with vLLM, for rl_hf --difficulty.

python -m deskmind_eyes.difficulty_vllm --model models/GUI-Owl-1.5-4B-Instruct --prompt-style tool \
    --data showui_desktop,osatlas_full_v3 --data-dir data/showui_desktop --osatlas-dir data/osatlas_desktop \
    --out runs/difficulty/guiowl_tool.jsonl

Same idea as difficulty.py (Tinker): `samples` answers per record at the RL temperature and resolution, count in-bbox
hits. Groups the model always (or never) solves give GRPO no gradient, so a new base needs its own scores. Records
and keys come from rl_hf's loaders; output lines are {"key", "source", "passes", "samples"}. Resumable.
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from deskmind_eyes.common import MAX_PIXELS, grounding_messages, parse_point, resize_to_max_pixels
from deskmind_eyes.rl_hf import in_bbox, load_pools


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="showui_desktop")
    ap.add_argument("--data-dir")
    ap.add_argument("--osatlas-dir")
    ap.add_argument("--groundcua-records", default="data/groundcua/records_functional.jsonl")
    ap.add_argument("--prompt-style", choices=["point", "tool", "owl"], default="point")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=MAX_PIXELS, help="RL's training resolution")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-model-len", type=int, default=8192)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    recs = load_pools(args.data, args, 0)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in open(out)} if out.exists() else set()
    todo = [r for r in recs if r["key"] not in done][: args.limit]
    print(f"{len(recs)} records, {len(done)} scored, {len(todo)} to do", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, limit_mm_per_prompt={"image": 1},
              gpu_memory_utilization=0.9)
    params = SamplingParams(n=args.samples, temperature=args.temperature, top_p=1.0,
                            max_tokens=48 if args.prompt_style == "point" else 128)
    load = lambda r: resize_to_max_pixels(Image.open(r["img_path"]).convert("RGB"), args.max_pixels)
    t0, hist = time.time(), [0] * (args.samples + 1)
    with ThreadPoolExecutor(8) as pool, open(out, "a") as f:
        for i in range(0, len(todo), args.batch):
            batch = todo[i:i + args.batch]
            convs = [grounding_messages(r["instruction"], args.prompt_style, {"type": "image_pil", "image_pil": img})
                     for r, img in zip(batch, pool.map(load, batch))]
            outs = llm.chat(convs, params, use_tqdm=False, chat_template_content_format="openai",
                            chat_template_kwargs={"enable_thinking": False})
            for r, o in zip(batch, outs):
                passes = sum(in_bbox(parse_point(c.text), r["bbox"]) for c in o.outputs)
                hist[passes] += 1
                f.write(json.dumps({"key": r["key"], "source": r["source"], "passes": passes,
                                    "samples": args.samples}, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i + len(batch)}/{len(todo)}] {time.time() - t0:.0f}s pass-count histogram {hist}", flush=True)


if __name__ == "__main__":
    main()
