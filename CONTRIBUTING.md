# Contributing to DeskMind Eyes · [中文](CONTRIBUTING.zh-CN.md)

Thanks for helping. DeskMind Eyes is small, and every change is judged the same way: does it put the point on the
right element more often, on a comparison someone else can repeat? The org-wide guidelines
([deskmind-ai/.github](https://github.com/deskmind-ai/.github)) apply here too, including the
[code of conduct](https://github.com/deskmind-ai/.github/blob/main/CODE_OF_CONDUCT.md).

## Set up

```bash
git clone https://github.com/deskmind-ai/eyes && cd eyes
uv sync --extra test             # Python 3.12; add --extra mlx on Apple Silicon
uv run pytest -q                 # must pass before you open a PR
```

- The tests need no GPU, no model and no dataset. They check that the leaderboard adapter implements exactly our
  prompt, resize and parsing (`tests/test_leaderboard_adapter.py`).
- Evaluating and training need a CUDA GPU with vLLM 0.19 (`uv pip install "vllm==0.19.*"`) and the `hf` extra
  (`uv sync --extra hf`); the earlier Tinker recipe needs the `tinker` extra. The MLX paths (`eval_mlx`,
  `ground_server`) run on an Apple Silicon Mac.

## Reproducing an eval

```bash
uv run python -m deskmind_eyes.prepare_data --only screenspot_pro        # downloads from the source, verifies images
uv run python -m deskmind_eyes.eval_vllm --model MODEL --benchmark screenspot_pro --prompt-style tool --out runs/eval/a.jsonl
uv run python -m deskmind_eyes.compare runs/eval/base.jsonl runs/eval/a.jsonl   # paired, McNemar's z
```

- **Compare only within one setup.** The same weights move by 0.5–0.9 points on ScreenSpot-Pro across GPUs and
  inference stacks. Run the baseline yourself, on your machine, with the same prompt style and resolution.
- **Report the paired numbers:** items gained, items lost, and z. We call |z| ≥ 2 significant and report anything
  smaller without a conclusion.
- Per-sample outputs (`runs/eval/*.jsonl`) are never committed. `results/` holds summary tables only.

## Reproducing training

[docs/training.md](docs/training.md) has every command of the three RL rounds, from difficulty scoring to merging
and serving. Each round runs on one GPU. Two things are easy to get wrong:

- **Image transport.** Screenshots must reach the model losslessly. `deskmind_eyes/image_encoding.py` exists because
  JPEG compression on the way to the sampler cost 5 points on Pro.
- **Model-written training data.** If a PR adds instructions or labels written by a model, say in the PR which model
  wrote them and under what terms.

A training PR states the base model, the data pools and their sizes, the steps, the hardware, and the eval before and
after on the same setup.

## Adding an eval set

1. **Download.** Add an entry to `SOURCES` in `deskmind_eyes/prepare_data.py`: the Hugging Face dataset repo, the
   local directory, the metadata files and how to list the images (or an `archive` when images ship as one zip).
2. **Load.** Add the name to `BENCHMARKS` in `deskmind_eyes/common.py` and a branch in `load_benchmark` that returns
   samples in the shared schema: `img_path`, `img_filename`, `instruction`, `bbox_xyxy` (pixels), `group`, `data_type`.
   `eval_vllm`, `eval_mlx` and `compare` pick it up from there.
3. **Check.** Run the base model on it, and compare the item count and the base accuracy with what the dataset's
   authors report. Put both in the PR.
4. **Document.** Add a section for the set to [docs/results.md](docs/results.md) and its 中文 version, with the
   dataset's licence and a link to its source.

## Dataset licences

- **Datasets are downloaded at runtime, never redistributed.** Do not commit images, annotations or any file derived
  from them (crops, rewritten instructions, difficulty scores) unless the dataset's licence allows it. `data/`,
  `runs/` and `models/` are git-ignored for this reason.
- **Respect each licence.** Before adding a source, read its licence and terms of use and name them in the PR. A set
  whose licence forbids the use, or that has no licence, is not added.
- **Say where contamination is possible.** If a training source shares apps or screenshots with an eval set (as
  GroundCUA and UI-Vision do), say so next to the number.
- The same goes for base models and weights: link to them, never re-upload them here.

## Pull requests

- **One change per PR,** with a short description of what changed and how you checked it.
- **Accuracy claims need paired evidence.** Paste the `compare` output against the baseline on the same setup, with
  the benchmark, the prompt style, the resolution and the hardware.
- **Keep the output formats.** `ground_server` answers `{"points": {name: [x, y] | null}}` in 0–1 relative
  coordinates, and DeskMind Hands depends on that shape. Model output is 0–1000 relative points. Extend both
  compatibly.
- **Style.** Match the surrounding code: small functions, comments that explain why, no new frameworks.
- **Checklist:**
  - [ ] `uv run pytest -q` passes;
  - [ ] no dataset files, per-sample outputs, weights, screenshots of your own apps or credentials in the diff;
  - [ ] new data sources name their licence;
  - [ ] docs updated in English and 中文 (`docs/*.md` and `docs/*.zh-CN.md`).

## Reporting problems

- **Bugs:** give the command, the model, the benchmark and the hardware, and a few sample ids that show the problem.
- **Security issues:** don't open a public issue. See
  [SECURITY.md](https://github.com/deskmind-ai/.github/blob/main/SECURITY.md).

By contributing you agree that your contribution is licensed under Apache-2.0, the licence of this repository.
Datasets and base models keep their own licences.
