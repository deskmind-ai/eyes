# Training · [中文](training.zh-CN.md)

Eyes-4B is GUI-Owl-1.5-4B-Instruct plus a LoRA trained with reinforcement learning on one GPU. This page is the recipe
behind the numbers in [results.md](results.md), and the lessons from an earlier phase on Qwen3.5-4B.

## Task and reward

- **Input:** a screenshot and a one-line instruction.
- **Output:** one click in 0–1000 relative coordinates, as a Qwen3-VL `computer_use` tool call (`--prompt-style tool`,
  GUI-Owl's native format). The earlier Qwen3.5 phase used a `{"point_2d": [x, y]}` prompt instead.
- **Reward:** 1 if the point lands in the target box, else 0.
- **Advantage:** reward minus the group mean. Groups where all answers get the same reward are skipped.

## RL setup (`deskmind_eyes.rl_hf`)

- **LoRA:** rank 32 on every language-model linear layer and on `lm_head`. The vision tower is frozen.
- **Sampling:** 8 answers per item at temperature 1.0 (no top-k / top-p), 32 items per step.
- **Optimizer:** AdamW(0.9, 0.95, eps 1e-8, no weight decay), lr 4e-5 decayed linearly to 10% at the last step.
- **Loss:** −Σ A · log p(token) per answer. Rollouts use HF `generate` on the policy itself, so no weight sync is
  needed.
- **Dynamic sampling** (`--dynamic-sampling`, DAPO-style): keep drawing items, up to 4 × 32 per step, until 32 groups
  have mixed rewards.

## Data

All data is downloaded from its source at runtime; none is redistributed here.

| pool | source | what we do with it |
|---|---|---|
| `showui_desktop` | showlab/ShowUI-desktop | as is (`prepare_data`) |
| `osatlas_full_v3` | OS-Atlas desktop (Windows / Linux / macOS) | full-resolution screenshots only; the short accessibility names are rewritten into ScreenSpot-Pro-style functional instructions by a frontier model (tool not included) |
| `groundcua` | ServiceNow/GroundCUA | functional instructions (`prepare_groundcua`) |

- **Why rewrite OS-Atlas instructions.** Raw OS-Atlas names ("Reload", two words on median) teach verbatim text
  matching. ScreenSpot-Pro instead asks for intents ("reload CMake cache").
  - A frontier model was shown the whole screen with numbered boxes, plus a zoomed crop per element, and asked for
    one instruction per box. The rewriting tool is not part of this repository.
  - **Disclosure:** its prompt used a handful of ScreenSpot-Pro test instructions, text only, as style examples (no
    screenshots, boxes or answers). The rewritten data was used for training, never for evaluation.
- **Contamination note.** GroundCUA shares its apps with UI-Vision. `dedup_uivision` finds near-duplicate screenshots
  between the two sets. The final model was trained without that filter, so its UI-Vision number is not clean.

## Difficulty pools

A group of 8 answers that are all right or all wrong gives no gradient. We therefore score every item by the model's
own pass rate first, then train on the items it only sometimes gets right:

```bash
python -m deskmind_eyes.difficulty_vllm --model models/GUI-Owl-1.5-4B-Instruct --prompt-style tool \
    --data showui_desktop,osatlas_full_v3 --data-dir data/showui_desktop --osatlas-dir data/osatlas_desktop \
    --out runs/difficulty/guiowl_tool.jsonl
```

`rl_hf --difficulty FILE --min-pass A --max-pass B` keeps items whose pass rate lies in [A, B].

## The three rounds

```bash
M=models/GUI-Owl-1.5-4B-Instruct
DATA="--data showui_desktop,osatlas_full_v3 --data-dir data/showui_desktop --osatlas-dir data/osatlas_desktop"

# 1. GRPO on the difficulty pool (pass rate 0.1-0.9, ~5.4K items): Pro 64.8 -> 65.8
python -m deskmind_eyes.rl_hf --model $M --prompt-style tool --max-tokens 64 $DATA \
    --difficulty runs/difficulty/guiowl_tool.jsonl --max-steps 40 --out runs/eyes-grpo

# 2. Same pool, dynamic sampling, again from the base: 66.7
python -m deskmind_eyes.rl_hf --model $M --prompt-style tool --max-tokens 64 $DATA \
    --difficulty runs/difficulty/guiowl_tool.jsonl --dynamic-sampling --max-steps 40 --out runs/eyes-dapo

# 3. Continue from round 2 on a harder, fresher GroundCUA pool at 2.5 MP (pass rate 1/8-4/8, ~3.3K items): 67.7
python -m deskmind_eyes.difficulty_vllm --model $M --prompt-style tool --data groundcua --max-pixels 2500000 \
    --out runs/difficulty/groundcua_tool_2p5m.jsonl
python -m deskmind_eyes.rl_hf --model $M --prompt-style tool --max-tokens 64 --data groundcua \
    --difficulty runs/difficulty/groundcua_tool_2p5m.jsonl --min-pass 0.1 --max-pass 0.5 --max-pixels 2500000 \
    --init-adapter runs/eyes-dapo/final --dynamic-sampling --max-steps 40 --save-every 10 --out runs/eyes-groundcua
```

**Why the third round uses a tighter pool.** After round 2, the policy already solved 61% of the groups in a
[1/8, 7/8] GroundCUA pool at step 0. Keeping only items the base solves at most half the time brought mixed groups back.

**Merge, evaluate, serve:**

```bash
python -m deskmind_eyes.merge_hf_lora --base $M --adapter runs/eyes-groundcua/final --out runs/merged/eyes-4b
python -m deskmind_eyes.eval_vllm --model runs/merged/eyes-4b --benchmark screenspot_pro --prompt-style tool
python -m mlx_vlm.convert --hf-path runs/merged/eyes-4b --mlx-path models/eyes-4b-mlx -q --q-bits 4   # on the Mac
```

`merge_hf_lora` takes several `--adapter` directories to average them. In our runs that gave at most +0.1.

## Earlier phase: Qwen3.5-4B on Tinker

`deskmind_eyes.train` (SFT) and `deskmind_eyes.rl` run on [Tinker](https://thinkingmachines.ai/tinker/). You need
`TINKER_API_KEY` in the environment. They use the same algorithm as `rl_hf`. `train_hf` is the single-GPU SFT
counterpart. Configs and per-step metrics of these runs are in `results/training/`.

```bash
python -m deskmind_eyes.eval_tinker benchmark=screenspot_pro native=True
python -m deskmind_eyes.difficulty sources=showui_desktop out=runs/difficulty/showui_desktop.jsonl
python -m deskmind_eyes.rl difficulty_file=runs/difficulty/showui_desktop.jsonl min_pass_rate=0.1 max_pass_rate=0.9 \
    eval_benchmarks=screenspot_v2,screenspot_pro eval_native=True max_steps=20 eval_every=10
```

## Lessons that cost us a training round each

- **The base model matters most.** Switching bases gained more than all our training on the old one.
- **RL is limited by the question pool, not the algorithm.** Every gain came from giving the model items it gets right
  only sometimes: difficulty filtering, dynamic sampling, re-scoring with the current policy, a new data source. A
  static filter goes stale fast. On Qwen3.5 the share of groups with signal fell from 45% to 6% in 40 steps.
- **Check how images reach the model.** Our first harness JPEG-compressed screenshots on the way to the sampler.
  That hid 5 points of ScreenSpot-Pro accuracy, and the training images went through the same
  compression.
- **Gains can be tied to the output format.** RL learned with the `point_2d` prompt kept only a third of its
  ScreenSpot-Pro gain under a tool-call prompt. Train in the format you will serve.
- **Higher training resolution is not automatically better.** On Qwen3.5, training at native resolution instead of
  ~1M pixels did not help (Pro 55.5 vs 56.9, not significant). More groups came back all-right, so fewer hard items
  contributed gradient.
- **Compare only within one platform.** The same checkpoint moved by up to 0.9 on Pro across GPUs and inference
  stacks, which is as large as some training gains.
