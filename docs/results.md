# Results · [中文](results.zh-CN.md)

All numbers are our own runs. The tables in [`results/eval_summary.md`](../results/eval_summary.md) and
[`results/eval_summary.csv`](../results/eval_summary.csv) are generated from our per-sample outputs and cover every run
quoted here.

## Protocol

- **Setup.** A single GPU, vLLM 0.19, bf16, greedy decoding, screenshots at native resolution (`eval_vllm`).
- **Prompt.** The Qwen3-VL `computer_use` tool prompt (`--prompt-style tool`). The answer is a `left_click` with a
  0–1000 coordinate.
- **Scoring.** A hit means the point lands inside the target box.
- **Zoom 0.5** (`--zoom 0.5`) is a second pass, the setting KV-Ground and GUI-Owl report as "zoom". Crop half the
  width and height around the first prediction (clipped at the borders), resize the crop to the full image size,
  predict again and map the point back.
- **Paired tests.** Each comparison is paired on the same items. `+` / `−` counts the items only one of the two models
  gets right; z = (+ − −) / √(+ + −), McNemar's statistic. We call |z| ≥ 2 significant.
- **Platform variance is real.** The same weights on another GPU and inference stack moved by 0.5–0.9 on
  ScreenSpot-Pro; on the lower-resolution ScreenSpot-v2 we saw no difference. We compare only within one setup.

## Eyes-4B

### ScreenSpot-Pro, full set (1,581 items)

| no zoom | CAD text | CAD icon | Creative text | Creative icon | Dev text | Dev icon | OS text | OS icon | Office text | Office icon | Scientific text | Scientific icon | **avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GUI-Owl-1.5-4B base | 59.4 | 39.1 | 72.7 | 41.3 | 82.5 | 51.0 | 81.3 | 49.4 | 85.9 | 49.1 | 85.4 | 41.8 | 64.8 |
| KV-Ground-4B | 57.4 | 39.1 | 77.3 | 46.9 | 81.8 | 49.0 | 74.8 | 51.7 | 88.1 | 54.7 | 88.2 | 47.3 | 66.1 |
| **Eyes-4B** | 62.4 | 43.8 | 77.3 | 44.8 | 84.4 | 51.7 | 80.4 | 51.7 | 89.3 | 56.6 | 88.2 | 46.4 | **67.7** |

| zoom 0.5 | CAD text | CAD icon | Creative text | Creative icon | Dev text | Dev icon | OS text | OS icon | Office text | Office icon | Scientific text | Scientific icon | **avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GUI-Owl-1.5-4B base | 83.8 | 59.4 | 81.8 | 56.6 | 87.7 | 59.3 | 79.4 | 66.3 | 92.1 | 75.5 | 89.6 | 55.5 | 76.2 |
| **Eyes-4B** | 82.2 | 59.4 | 83.8 | 56.6 | 88.3 | 62.8 | 83.2 | 66.3 | 93.2 | 77.4 | 91.0 | 60.9 | **77.5** |

| paired | + | − | z |
|---|---|---|---|
| Eyes-4B vs base, no zoom | 81 | 34 | 4.38 |
| Eyes-4B vs KV-Ground-4B, no zoom | 68 | 42 | 2.48 |
| Eyes-4B vs base, zoom 0.5 | 74 | 52 | 1.96 |

- **Complementary errors.** KV-Ground-4B is stronger on Creative icons; we are stronger on text targets and on larger
  targets. If either model's answer counted, Pro would be 70.4, which suggests that small, high-resolution targets
  are where more data would help.
- **Self-reported numbers from other harnesses:**
  - KV-Ground-4B: 67.0 (our setup: 66.1);
  - GUI-Owl-1.5-4B: 66.8;
  - Qwen-UI-Agent-4B: 67.8 (weights not released, so we could not run it).

### As the Mac app runs it (4-bit MLX, ≤2 MP)

The same 1,581 items with the weights and settings the DeskMind app uses: `deskmind/eyes-4b` at the `mlx-4bit`
revision (de8e13b), mlx-vlm 0.7.4 from the app's own runtime, one pass, screenshots scaled to at most 2,000,000
pixels (`ground_server`'s default), greedy decoding, the point prompt. Apple M4 Pro, 48 GB.

| | CAD text | CAD icon | Creative text | Creative icon | Dev text | Dev icon | OS text | OS icon | Office text | Office icon | Scientific text | Scientific icon | **avg** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Eyes-4B, GPU bf16, native (above) | 62.4 | 43.8 | 77.3 | 44.8 | 84.4 | 51.7 | 80.4 | 51.7 | 89.3 | 56.6 | 88.2 | 46.4 | 67.7 |
| **Eyes-4B, Mac app: 4-bit MLX, ≤2 MP** | 38.6 | 23.4 | 64.6 | 21.7 | 74.0 | 26.2 | 65.4 | 30.3 | 75.1 | 41.5 | 77.8 | 34.5 | **50.9** |

- Text targets 64.8, icon targets 28.3. Every answer parsed.
- The loss is mostly icons (−20 to −28 points per group). ScreenSpot-Pro screenshots are mostly 4K-class, so 2 MP
  is about a fourfold downscale and small icons lose the most. How much is resolution and how much quantization is
  being measured.
- Latency per query: p50 4.7 s, p95 7.8 s (about 2,000 prompt tokens); peak MLX memory 4.5 GB. Part of the run shared
  the Mac with another model server, so take the times as an upper bound.
- Command: `python -m deskmind_eyes.eval_mlx <model> out.jsonl --benchmark screenspot_pro --max-pixels 2000000`.

### Smallest targets (Pro, no zoom)

| | quarter of targets with the smallest boxes | Creative icon |
|---|---|---|
| KV-Ground-4B | 46.6 | 46.9 |
| **Eyes-4B** | 46.1 | 44.8 |

Small targets are Eyes-4B's known gap: KV-Ground is level on the smallest quarter and ahead on Creative-app icons.

### ScreenSpot-v2 (1,272 items) and UI-Vision (5,479 items)

| ScreenSpot-v2 | mobile text | mobile icon | desktop text | desktop icon | web text | web icon | **avg** |
|---|---|---|---|---|---|---|---|
| GUI-Owl-1.5-4B base | 98.3 | 89.1 | 95.4 | 85.0 | 95.7 | 88.7 | 92.8 |
| **Eyes-4B** | 99.3 | 91.5 | 97.4 | 88.6 | 97.0 | 92.1 | **95.0** |

Paired: +30 / −3, z = 4.7.

| UI-Vision element grounding | basic text | basic icon | functional text | functional icon | spatial text | spatial icon | **avg** |
|---|---|---|---|---|---|---|---|
| GUI-Owl-1.5-4B base | 66.9 | 32.7 | 58.5 | 31.2 | 45.9 | 12.6 | 30.7 |
| Eyes-4B-DAPO (no GroundCUA) | 72.0 | 33.8 | 64.2 | 33.7 | 45.9 | 13.1 | **32.5** |
| Eyes-4B (final) ¹ | 73.9 | 34.6 | 68.4 | 33.8 | 47.3 | 12.8 | 33.1 |

Paired, DAPO vs base: +228 / −128, z = 5.3.

¹ Not a clean number. All of UI-Vision's apps also appear in GroundCUA, which the final round trained on, and the
GroundCUA paper does not mention removing duplicate screenshots. ScreenSpot-Pro (commercial apps) and
ScreenSpot-v2 are not affected. `deskmind_eyes.dedup_uivision` finds near-duplicate screenshots between the two sets.

### The path from 64.8 to 67.7 (Pro, no zoom)

| step | Pro | vs previous |
|---|---|---|
| GUI-Owl-1.5-4B-Instruct, no training | 64.8 | |
| GRPO, 40 steps, 5.4K-item difficulty pool | 65.8 | +1.0 |
| DAPO-style dynamic sampling instead, 40 steps, same pool | 66.7 | +0.9 |
| continue with dynamic sampling on a GroundCUA pool, 2.5 MP, 40 steps | **67.7** | +1.0 |

The GRPO and DAPO rounds each start from the base; the GroundCUA round continues from the DAPO adapter. Details:
[training.md](training.md).

**Tried, no gain:**
- **Averaging checkpoint weights:** 66.9 vs 66.7 for DAPO, 67.6 vs 67.7 for GroundCUA.
- **GUI-Owl's own evaluation prompt** (its system prompt, images capped at 9,800 visual tokens): the base drops to
  63.2 on our stack. We could not reproduce GUI-Owl's self-reported 66.8 with it.

## Earlier phase: Qwen3.5-4B on Tinker

Before switching base models we trained Qwen3.5-4B with SFT and RL on [Tinker](https://thinkingmachines.ai/tinker/).
These numbers use our `point_2d` prompt and Tinker sampling. Compare them only with each other.

| Pro, native resolution | JPEG transport | lossless transport |
|---|---|---|
| Qwen3.5-4B base | 52.9 | 58.1 |
| RL, difficulty-filtered pool, step 20 | 59.8 | 61.9 |
| RL, dynamic sampling, step 30 | 59.6 | 63.2 |

- **Image transport matters.** Our first harness sent screenshots as quality-75 JPEG, which cost the base 5 points on
  Pro. `deskmind_eyes.image_encoding` sends PNG or high-quality 4:4:4 JPEG instead. ScreenSpot-v2 was barely
  affected (91.3 → 91.1).
- **ScreenSpot-v2, JPEG transport:** base 91.3; filtered RL 93.5; dynamic sampling 94.0.
- **Gains are format-bound.** Under the community ScreenSpot-Pro adapter (a tool-call prompt), the filtered RL
  checkpoint gains only +2.2 over the base (49.0 → 51.1), against +6.9 under the prompt it was trained with.
- **Base model matters most.** Our best Qwen3.5-4B run (63.2, Tinker) stayed below GUI-Owl-1.5-4B with no training
  (64.8, vLLM). The harnesses differ, but switching bases was clearly the largest single step.

## Files

- `results/eval_summary.md`, `results/eval_summary.csv`: every eval run, by suite, with per-category accuracy and the
  paired comparisons above.
- `results/training/<run>/`: config and per-step metrics (`metrics.csv`) of the Qwen3.5-4B Tinker runs.
