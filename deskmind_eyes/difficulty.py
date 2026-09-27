"""Score training records by how often the base model already gets them right.

For each record we sample `samples` answers at the RL temperature and resolution and
count in-bbox hits. RL only learns from groups with mixed outcomes, so records the base
model always solves (pass rate 1.0) add no gradient; rl.py can drop them with
`difficulty_file=... max_pass_rate=0.875`.

python -m deskmind_eyes.difficulty sources=showui_desktop out=runs/difficulty/desktop.jsonl
"""

import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import chz
import tinker
from tinker import types

from deskmind_eyes.common import MAX_PIXELS, NATIVE_MAX_PIXELS, PROMPT, parse_point
from deskmind_eyes.data import load_image, load_train_records
from deskmind_eyes.rl import in_bbox
from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import ImagePart, Message, TextPart, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer


@chz.chz
class Config:
    sources: str = "showui_desktop"
    out: str = "runs/difficulty/showui_desktop.jsonl"
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5_disable_thinking"
    model_path: str | None = None  # score against a checkpoint instead of the base model
    samples: int = 8
    temperature: float = 1.0
    native: bool = False  # match the resolution RL will train at
    max_parallel: int = 24  # each request uploads a screenshot; 64 saturated the uplink
    limit: int | None = None


async def main(cfg: Config):
    renderer = renderers.get_renderer(
        cfg.renderer_name, get_tokenizer(cfg.model_name),
        image_processor=get_image_processor(cfg.model_name),
    )
    service = tinker.ServiceClient()
    client = (service.create_sampling_client(model_path=cfg.model_path) if cfg.model_path
              else service.create_sampling_client(base_model=cfg.model_name))
    params = types.SamplingParams(max_tokens=48, temperature=cfg.temperature,
                                  stop=renderer.get_stop_sequences())
    max_pixels = NATIVE_MAX_PIXELS if cfg.native else MAX_PIXELS

    records = load_train_records(cfg.sources)
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in open(out)} if out.exists() else set()
    todo = [r for r in records if r["key"] not in done][: cfg.limit]
    print(f"{len(records)} records, {len(done)} already scored, {len(todo)} to score", flush=True)

    sem = asyncio.Semaphore(cfg.max_parallel)
    t0 = time.time()
    n_done = 0

    async def score(rec: dict, f):
        nonlocal n_done
        async with sem:
            prompt = renderer.build_generation_prompt([Message(role="user", content=[
                ImagePart(type="image", image=load_image(rec["img_path"], max_pixels)),
                TextPart(type="text", text=PROMPT.format(instruction=rec["instruction"])),
            ])])
            res = await client.sample_async(prompt=prompt, num_samples=cfg.samples, sampling_params=params)
        passes = sum(
            in_bbox(parse_point(get_text_content(renderer.parse_response(s.tokens)[0])), rec["bbox"])
            for s in res.sequences
        )
        f.write(json.dumps({"key": rec["key"], "source": rec["source"], "passes": passes,
                            "samples": cfg.samples}, ensure_ascii=False) + "\n")
        n_done += 1
        if n_done % 500 == 0:
            f.flush()
            print(f"{n_done}/{len(todo)} scored, {(time.time() - t0) / 60:.1f} min", flush=True)

    with open(out, "a") as f:
        await asyncio.gather(*[score(r, f) for r in todo])

    rows = [json.loads(l) for l in open(out)]
    for src in sorted({r["source"] for r in rows}):
        hist = Counter(r["passes"] for r in rows if r["source"] == src)
        n = sum(hist.values())
        print(f"[{src}] n={n} pass-count histogram: "
              + " ".join(f"{k}:{hist[k] / n:.0%}" for k in range(cfg.samples + 1))
              + f" | mean pass rate {sum(k * v for k, v in hist.items()) / n / cfg.samples:.3f}")


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
