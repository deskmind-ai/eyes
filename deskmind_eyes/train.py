"""Grounding SFT of Qwen3.5-4B on ShowUI-desktop with Tinker (LoRA), evaluated on ScreenSpot.

python -m deskmind_eyes.train                      # full run (1 epoch)
python -m deskmind_eyes.train max_steps=2 log_path=runs/smoke   # smoke test
"""

import asyncio

import chz

from deskmind_eyes.data import RecordsBuilder
from deskmind_eyes.evaluator import ScreenSpotEvaluatorBuilder
from tinker_cookbook import cli_utils
from tinker_cookbook.hyperparam_utils import get_lr
from tinker_cookbook.supervised import train


@chz.chz
class Config:
    log_path: str = "runs/sft-showui-desktop"
    train_sources: str = "showui_desktop"  # comma-separated, see data.TRAIN_SOURCES
    model_name: str = "Qwen/Qwen3.5-4B"
    renderer_name: str = "qwen3_5_disable_thinking"
    learning_rate: float | None = None  # None -> cookbook's recommended LoRA LR
    lora_rank: int = 32
    batch_size: int = 32
    num_epochs: int = 1
    max_steps: int | None = None
    eval_every: int = 25  # held-out NLL
    screenspot_every: int = 25  # full ScreenSpot eval (~70s); dense enough to tell dips from trends
    screenspot_eval: bool = True  # False: no Tinker eval at all (evaluate checkpoints on your own GPU)
    save_every: int = 25  # match screenspot_every so the best-scoring step has a checkpoint
    wandb_project: str | None = None
    behavior_if_log_dir_exists: str = "ask"


def main(cfg: Config):
    cli_utils.check_log_dir(cfg.log_path, behavior_if_exists=cfg.behavior_if_log_dir_exists)
    train_cfg = train.Config(
        log_path=cfg.log_path,
        model_name=cfg.model_name,
        # cookbook tags the session with recipe_name only
        recipe_name="deskmind_eyes_sft",
        renderer_name=cfg.renderer_name,
        dataset_builder=RecordsBuilder(
            model_name=cfg.model_name, renderer_name=cfg.renderer_name, batch_size=cfg.batch_size,
            sources=cfg.train_sources,
        ),
        infrequent_evaluator_builders=[
            ScreenSpotEvaluatorBuilder(model_name=cfg.model_name, renderer_name=cfg.renderer_name)
        ] if cfg.screenspot_eval else [],
        learning_rate=cfg.learning_rate or get_lr(cfg.model_name),
        lr_schedule="linear",
        num_epochs=cfg.num_epochs,
        lora_rank=cfg.lora_rank,
        eval_every=cfg.eval_every,
        infrequent_eval_every=cfg.screenspot_every,
        save_every=cfg.save_every,
        max_steps=cfg.max_steps,
        wandb_project=cfg.wandb_project,
    )
    asyncio.run(train.main(train_cfg))


if __name__ == "__main__":
    main(chz.entrypoint(Config))
