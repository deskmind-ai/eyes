"""Evaluate a base model or a Tinker checkpoint on ScreenSpot v1 / V2 / Pro via Tinker sampling.

python -m deskmind_eyes.eval_tinker                                          # base, ScreenSpot v1, 1M px
python -m deskmind_eyes.eval_tinker benchmark=screenspot_v2 native=True      # full resolution
python -m deskmind_eyes.eval_tinker benchmark=screenspot_pro native=True
python -m deskmind_eyes.eval_tinker model_path=tinker://.../sampler_weights/final
python -m deskmind_eyes.eval_tinker n_eval=100                               # quick smoke test
"""

import asyncio
import json
import time

import chz
import tinker

from deskmind_eyes.common import MAX_PIXELS, NATIVE_MAX_PIXELS
from deskmind_eyes.evaluator import ScreenSpotEvaluatorBuilder


@chz.chz
class Config:
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5_disable_thinking"
    model_path: str | None = None  # tinker:// sampler weights; None = base model
    benchmark: str = "screenspot"  # screenspot | screenspot_v2 | screenspot_pro
    native: bool = False  # True: keep full resolution (up to the processor's 16.7M px ceiling)
    prompt_style: str = "point"  # point | bbox | short | tool
    max_tokens: int = 48  # raise (e.g. 2048) with renderer_name=qwen3_5 to allow thinking
    max_parallel: int = 64
    n_eval: int | None = None
    output_path: str | None = None  # default: runs/eval/<benchmark>_<native|1mp>_<model>.jsonl


async def main(cfg: Config):
    service = tinker.ServiceClient()
    if cfg.model_path:
        client = service.create_sampling_client(model_path=cfg.model_path)
    else:
        client = service.create_sampling_client(base_model=cfg.model_name)
    tag = (cfg.model_path or cfg.model_name).replace("tinker://", "").replace("/", "_").replace(":", "_")
    output_path = cfg.output_path or (
        f"runs/eval/{cfg.benchmark}_{'native' if cfg.native else '1mp'}_{cfg.prompt_style}_{cfg.renderer_name}_{tag}.jsonl"
    )
    evaluator = ScreenSpotEvaluatorBuilder(
        model_name=cfg.model_name, renderer_name=cfg.renderer_name, benchmark=cfg.benchmark,
        max_pixels=NATIVE_MAX_PIXELS if cfg.native else MAX_PIXELS,
        prompt_style=cfg.prompt_style, max_tokens=cfg.max_tokens, max_parallel=cfg.max_parallel, n_eval=cfg.n_eval, output_path=output_path,
    )()
    t = time.time()
    table, rows = await evaluator.run(client)
    print(json.dumps({
        **table, "benchmark": cfg.benchmark, "native": cfg.native, "prompt_style": cfg.prompt_style,
        "model": cfg.model_path or cfg.model_name, "elapsed_s": round(time.time() - t),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / len(rows)),
        "output_path": output_path,
    }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
