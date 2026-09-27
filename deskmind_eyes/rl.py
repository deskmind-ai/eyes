"""GRPO-style grounding RL on ShowUI-desktop with raw Tinker primitives.

Reward = 1 if the sampled click point lands inside the target bbox, else 0.
Per screenshot-instruction we sample `group_size` answers, center rewards within
the group (advantage = r - mean), and train with Tinker's `importance_sampling` loss.

python -m deskmind_eyes.rl max_steps=2 eval_every=0 log_path=runs/rl-smoke   # smoke
python -m deskmind_eyes.rl                                                                # full (1M px, eval v1)
python -m deskmind_eyes.rl native=True eval_native=True eval_benchmarks=screenspot_v2,screenspot_pro   # high-res
"""

import asyncio
import json
import logging
import random
import time
from pathlib import Path

import chz
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData

from deskmind_eyes.common import MAX_PIXELS, NATIVE_MAX_PIXELS, PROMPT, parse_point
from deskmind_eyes.data import load_image, load_train_records
from deskmind_eyes.evaluator import ScreenSpotEvaluatorBuilder
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.renderers import ImagePart, Message, TextPart, get_text_content
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.utils import ml_log

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    log_path: str = "runs/rl-showui-desktop"
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5_disable_thinking"
    init_state_path: str | None = None  # e.g. an SFT tinker://.../weights/000050; None = base
    lora_rank: int = 32
    batch_size: int = 32  # screenshot-instruction prompts per step
    group_size: int = 8  # samples per prompt
    learning_rate: float = 4e-5
    # "constant" or "linear" (decays to lr_end_frac * learning_rate at max_steps). Late steps have
    # few groups with signal and are noisy; two identical runs diverged by 4.8 Pro points there.
    lr_schedule: str = "constant"
    lr_end_frac: float = 0.1
    temperature: float = 1.0
    max_tokens: int = 48
    max_steps: int = 50
    train_sources: str = "showui_desktop"  # comma-separated: showui_desktop,showui_web
    difficulty_file: str | None = None  # jsonl from difficulty.py; None = no filtering
    min_pass_rate: float = 0.0  # keep records whose base pass rate is in [min, max]
    max_pass_rate: float = 1.0
    # DAPO-style dynamic sampling: keep drawing batches until `batch_size` groups with mixed
    # rewards (under the *current* policy) are collected, up to max_sampling_rounds batches.
    dynamic_sampling: bool = False
    max_sampling_rounds: int = 3
    fwd_bwd_chunk: int = 64  # datums per forward_backward request (gradients accumulate)
    native: bool = False  # train on full-resolution screenshots instead of ~1M px
    # Comma-separated benchmarks evaluated at eval_every and at the end (see common.BENCHMARKS).
    eval_benchmarks: str = "screenspot"
    eval_native: bool = False  # evaluate at full resolution (needed for screenspot_pro)
    eval_every: int = 10  # 0 disables periodic eval (a final eval always runs)
    eval_n: int | None = None  # fixed random subset (seed 0) for periodic evals; final eval uses all
    save_every: int = 10
    seed: int = 0
    wandb_project: str | None = None


def in_bbox(point, bbox) -> bool:
    """point: relative (x, y); bbox: relative [x1, y1, x2, y2] (ShowUI format)."""
    if point is None:
        return False
    x1, y1, x2, y2 = bbox
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def filter_by_difficulty(records: list[dict], difficulty_file: str, min_pass: float, max_pass: float) -> list[dict]:
    """Keep records whose base-model pass rate (from difficulty.py) is within [min_pass, max_pass]."""
    rate = {}
    for line in open(difficulty_file):
        d = json.loads(line)
        rate[d["key"]] = d["passes"] / d["samples"]
    kept = [r for r in records if r["key"] in rate and min_pass <= rate[r["key"]] <= max_pass]
    logger.info(f"difficulty filter [{min_pass}, {max_pass}]: {len(kept)} / {len(records)} records kept "
                f"({sum(r['key'] in rate for r in records)} scored)")
    return kept


def build_datums(prompt: types.ModelInput, seqs, advantages) -> list[types.Datum]:
    datums = []
    ob_len = prompt.length - 1
    for seq, adv in zip(seqs, advantages):
        model_input = prompt.append(types.EncodedTextChunk(tokens=seq.tokens[:-1]))
        n = model_input.length
        datums.append(types.Datum(
            model_input=model_input,
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor([0] * ob_len + seq.tokens)),
                "logprobs": TensorData.from_torch(torch.tensor([0.0] * ob_len + seq.logprobs)),
                "advantages": TensorData.from_torch(torch.tensor([0.0] * ob_len + [adv] * (n - ob_len))),
            },
        ))
    return datums


async def run_evals(evaluators: dict, sampling_client, eval_dir: Path, step: int, suffix: str = "") -> dict:
    """Run each benchmark, save per-sample rows to eval/<benchmark>_stepNNNN<suffix>.jsonl."""
    metrics = {}
    for name, evaluator in evaluators.items():
        t = time.time()
        table, rows = await evaluator.run(sampling_client)
        eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_dir / f"{name}_step{step:04d}{suffix}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        metrics.update({f"{name}{suffix}/{k}": float(v) for k, v in table.items()})
        metrics[f"time/eval_{name}"] = time.time() - t
        logger.info(f"step {step} {name} avg {table['avg']}")
    return metrics


async def main(cfg: Config):
    ml_logger = ml_log.setup_logging(
        log_dir=cfg.log_path, wandb_project=cfg.wandb_project, config=cfg,
        do_configure_logging_module=True,
    )
    log_path = Path(cfg.log_path)
    renderer = renderers.get_renderer(
        cfg.renderer_name, get_tokenizer(cfg.model_name),
        image_processor=get_image_processor(cfg.model_name),
    )
    train_max_pixels = NATIVE_MAX_PIXELS if cfg.native else MAX_PIXELS
    def make_evaluators(n_eval):
        return {
            name: ScreenSpotEvaluatorBuilder(
                model_name=cfg.model_name, renderer_name=cfg.renderer_name, benchmark=name,
                max_pixels=NATIVE_MAX_PIXELS if cfg.eval_native else MAX_PIXELS, max_parallel=16,
                n_eval=n_eval,
            )()
            for name in cfg.eval_benchmarks.split(",") if name
        }

    evaluators = make_evaluators(cfg.eval_n)  # periodic (subset when eval_n is set)
    final_evaluators = make_evaluators(None) if cfg.eval_n else evaluators
    records = load_train_records(cfg.train_sources, cfg.seed)
    if cfg.difficulty_file:
        records = filter_by_difficulty(records, cfg.difficulty_file, cfg.min_pass_rate, cfg.max_pass_rate)
    logger.info(f"{len(records)} train records; {len(records) // cfg.batch_size} steps per epoch")

    service = tinker.ServiceClient()
    resume = checkpoint_utils.get_last_checkpoint(cfg.log_path)
    if resume:
        tc = await service.create_training_client_from_state_with_optimizer_async(resume.state_path)
        start_step = resume.batch
        logger.info(f"Resumed from {resume.state_path} at step {start_step}")
    elif cfg.init_state_path:
        tc = await service.create_training_client_from_state_async(cfg.init_state_path)
        start_step = 0
    else:
        tc = await service.create_lora_training_client_async(base_model=cfg.model_name, rank=cfg.lora_rank)
        start_step = 0
    # Position in the shuffled record list. Without dynamic sampling this is step * batch_size;
    # with it, a step may consume several batches, so the cursor is saved in checkpoints.
    cursor = resume.extra.get("cursor", start_step * cfg.batch_size) if resume else 0

    sampling_params = types.SamplingParams(
        max_tokens=cfg.max_tokens, temperature=cfg.temperature, stop=renderer.get_stop_sequences(),
    )
    def lr_at(step: int) -> float:
        if cfg.lr_schedule == "linear":
            frac = step / max(1, cfg.max_steps - 1)
            return cfg.learning_rate * (1 - (1 - cfg.lr_end_frac) * frac)
        return cfg.learning_rate

    for step in range(start_step, cfg.max_steps):
        t0 = time.time()
        lr = lr_at(step)
        adam = types.AdamParams(learning_rate=lr, beta1=0.9, beta2=0.95, eps=1e-8)
        metrics: dict[str, float] = {"progress/step": step, "optim/lr": lr}
        sc = await tc.save_weights_and_get_sampling_client_async()

        if cfg.eval_every and step % cfg.eval_every == 0 and step > 0:
            metrics.update(await run_evals(evaluators, sc, log_path / "eval", step,
                                           suffix=f"_n{cfg.eval_n}" if cfg.eval_n else ""))

        if cfg.save_every and step % cfg.save_every == 0 and step > 0:
            await checkpoint_utils.save_checkpoint_async(
                tc, name=f"{step:06d}", log_path=cfg.log_path, kind="both",
                loop_state={"batch": step, "cursor": cursor},
            )

        # Sample rounds of `batch_size` prompts; keep groups whose rewards are not all equal.
        t = time.time()
        kept, rewards, n_prompts, rounds = [], [], 0, 0
        n_parse_fail = n_all_right = n_all_wrong = n_tokens = prompt_tokens = 0
        while True:
            batch = [records[(cursor + i) % len(records)] for i in range(cfg.batch_size)]
            cursor += cfg.batch_size
            rounds += 1
            prompts = [
                renderer.build_generation_prompt([Message(role="user", content=[
                    ImagePart(type="image", image=load_image(r["img_path"], train_max_pixels)),
                    TextPart(type="text", text=PROMPT.format(instruction=r["instruction"])),
                ])])
                for r in batch
            ]
            results = await asyncio.gather(*[
                sc.sample_async(prompt=p, num_samples=cfg.group_size, sampling_params=sampling_params)
                for p in prompts
            ])
            for rec, prompt, res in zip(batch, prompts, results):
                n_prompts += 1
                prompt_tokens += prompt.length
                group_rewards = []
                for seq in res.sequences:
                    point = parse_point(get_text_content(renderer.parse_response(seq.tokens)[0]))
                    n_parse_fail += point is None
                    n_tokens += len(seq.tokens)
                    group_rewards.append(float(in_bbox(point, rec["bbox"])))
                rewards.extend(group_rewards)
                mean = sum(group_rewards) / len(group_rewards)
                if mean == 1.0:
                    n_all_right += 1
                elif mean == 0.0:
                    n_all_wrong += 1
                else:
                    kept.append((prompt, res.sequences, [r - mean for r in group_rewards]))
            if (not cfg.dynamic_sampling or len(kept) >= cfg.batch_size
                    or rounds >= cfg.max_sampling_rounds):
                break
        if cfg.dynamic_sampling:
            kept = kept[: cfg.batch_size]
        datums = [d for prompt, seqs, advs in kept for d in build_datums(prompt, seqs, advs)]
        metrics["time/sample"] = time.time() - t

        n_samples = n_prompts * cfg.group_size
        metrics.update({
            "reward/mean": sum(rewards) / len(rewards),
            "groups/all_right_frac": n_all_right / n_prompts,
            "groups/all_wrong_frac": n_all_wrong / n_prompts,
            # groups actually trained on, relative to the target batch size
            "groups/trained_frac": len(kept) / cfg.batch_size,
            "sample/rounds": rounds,
            "sample/prompts": n_prompts,
            "sample/parse_fail_frac": n_parse_fail / n_samples,
            "sample/mean_tokens": n_tokens / n_samples,
            "sample/mean_prompt_tokens": prompt_tokens / n_prompts,
        })

        if datums:
            t = time.time()
            # Wait for each forward_backward before stepping: on a (rare, not reproducible)
            # non-finite loss the request fails and adds no gradient. Gradients accumulate across
            # forward_backward calls until optim_step, so datums go in chunks (each datum carries
            # its screenshot; one 256-datum request is ~40-100 MB and times out on slow uplinks).
            n_ok = 0
            for i in range(0, len(datums), cfg.fwd_bwd_chunk):
                chunk = datums[i : i + cfg.fwd_bwd_chunk]
                try:
                    await (await tc.forward_backward_async(chunk, loss_fn="importance_sampling")).result_async()
                    n_ok += len(chunk)
                except tinker.RequestFailedError as e:
                    if "Non-finite" not in str(e):
                        raise
                    logger.warning(f"step {step}: non-finite loss in a chunk of {len(chunk)} datums, dropped")
                    metrics["train/skipped_nonfinite"] = metrics.get("train/skipped_nonfinite", 0) + len(chunk)
            if n_ok:  # step with whatever valid gradient was accumulated
                opt_result = await (await tc.optim_step_async(adam)).result_async()
                if opt_result.metrics:
                    metrics.update(opt_result.metrics)
            metrics["time/train"] = time.time() - t
        else:
            logger.warning(f"step {step}: every group had identical rewards, skipping update")

        metrics["time/step"] = time.time() - t0
        ml_logger.log_metrics(metrics, step=step)

    sc = await tc.save_weights_and_get_sampling_client_async()
    final = await run_evals(final_evaluators, sc, log_path / "eval", cfg.max_steps)
    ml_logger.log_metrics(final, step=cfg.max_steps)
    await checkpoint_utils.save_checkpoint_async(
        tc, name="final", log_path=cfg.log_path, kind="both",
        loop_state={"batch": cfg.max_steps, "final": True, "cursor": cursor}, ttl_seconds=None,
    )
    ml_logger.close()


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
