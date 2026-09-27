"""Cross-check the leaderboard adapter's protocol on MLX against our Tinker eval, sample by sample.

Uses the adapter's own PROMPT / preprocess_image / parse_response (not deskmind_eyes's copies),
runs Qwen3.5-4B locally with mlx-vlm, and compares hit/miss and click points with the Tinker
base-model rows in runs/eval/pro/base_native_point.jsonl.

~/.local/share/uv/tools/mlx-vlm/bin/python -m leaderboard.verify_adapter_mlx ~/models/Qwen3.5-4B-bf16 --n 100
(~10-15 s per Pro screenshot on an M4 Pro; do not run alongside other heavy jobs)
"""

import argparse
import importlib.util
import json
import math
import random
import time
from pathlib import Path

import mlx.core as mx
from mlx_vlm import apply_chat_template, generate, load
from PIL import Image

from deskmind_eyes.common import load_benchmark, point_in_bbox

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("qwen3_5_point", ROOT / "leaderboard" / "models" / "qwen3_5_point.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--reference", default=str(ROOT / "runs/eval/pro/base_native_point.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs/eval/adapter_mlx_pro.jsonl"))
    args = ap.parse_args()

    key = lambda r: (r["img_filename"], r["instruction"])
    ref = {key(r): r for r in map(json.loads, open(args.reference))}
    samples = random.Random(0).sample(load_benchmark("screenspot_pro"), args.n)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {key(r): r for r in map(json.loads, open(out))} if out.exists() else {}
    todo = [s for s in samples if key(s) not in done]
    print(f"{len(samples)} samples, {len(done)} done, {len(todo)} to run", flush=True)

    if todo:
        mx.set_cache_limit(1024**3)
        model, processor = load(args.model)
        t0 = time.perf_counter()
        with open(out, "a") as f:
            for i, s in enumerate(todo):
                orig = Image.open(s["img_path"]).convert("RGB")
                prompt = apply_chat_template(
                    processor, model.config, adapter.PROMPT.format(instruction=s["instruction"]),
                    num_images=1, enable_thinking=False,
                )
                res = generate(model, processor, prompt, image=[adapter.preprocess_image(orig)],
                               max_tokens=adapter.MAX_NEW_TOKENS, temperature=0.0)
                point = adapter.parse_response(res.text)
                r = {"img_filename": s["img_filename"], "instruction": s["instruction"],
                     "group": s["group"], "data_type": s["data_type"], "point": point,
                     "hit": point_in_bbox(s, point, orig.size), "output": res.text,
                     "prompt_tokens": res.prompt_tokens, "peak_mem_gb": round(res.peak_memory, 2)}
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                f.flush()
                done[key(r)] = r
                if (i + 1) % 10 == 0 or i + 1 == len(todo):
                    el = time.perf_counter() - t0
                    print(f"[{i + 1}/{len(todo)}] {el / 60:.1f} min, eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min, "
                          f"peak {mx.get_peak_memory() / 1024**3:.1f} GB", flush=True)

    rows = [done[key(s)] for s in samples]
    agree = sum(r["hit"] == ref[key(r)]["hit"] for r in rows)
    dists = [math.dist(r["point"], ref[key(r)]["point"]) * 1000
             for r in rows if r["point"] and ref[key(r)]["point"]]
    exact = sum(d < 0.5 for d in dists)
    print(json.dumps({
        "n": len(rows),
        "mlx_acc": round(100 * sum(r["hit"] for r in rows) / len(rows), 1),
        "tinker_acc_same_samples": round(100 * sum(ref[key(r)]["hit"] for r in rows) / len(rows), 1),
        "hit_agreement": f"{agree}/{len(rows)}",
        "same_point(<0.5/1000)": f"{exact}/{len(dists)}",
        "median_point_dist_per_1000": round(sorted(dists)[len(dists) // 2], 1) if dists else None,
        "parse_fail": sum(r["point"] is None for r in rows),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
