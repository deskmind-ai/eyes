# ScreenSpot-Pro leaderboard adapter

`models/qwen3_5_point.py` implements our `point_2d` protocol in the format of the official
[ScreenSpot-Pro-GUI-Grounding](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding) evaluation script. It is
the protocol of the earlier Qwen3.5-4B phase. The Eyes-4B numbers use the `computer_use` tool prompt through
`deskmind_eyes.eval_vllm --prompt-style tool` instead.

## Protocol

| item | setting |
|---|---|
| Prompt | `Locate the UI element ... {"point_2d": [x, y]} ... 0-1000 ...\nDescription: {instruction}` (one user message: image + text, no system message) |
| Thinking | off (`enable_thinking=False`) |
| Resolution | native; only images above 16,777,216 pixels are downscaled (LANCZOS), the rest is left to the Qwen3.5 processor |
| Decoding | greedy, `max_new_tokens=48` |
| Parsing | first `[x, y]`, 0–1000 relative (0–1 decimals also accepted) |

Consistency with `deskmind_eyes/common.py`: `python -m pytest tests/test_leaderboard_adapter.py`.
Sample-by-sample parity with the Tinker eval on a local MLX subset: `python -m leaderboard.verify_adapter_mlx`
(pass a Tinker eval output with `--reference`).

## Running it in the official repository

```bash
git clone https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding && cd ScreenSpot-Pro-GUI-Grounding
cp /path/to/eyes/leaderboard/models/qwen3_5_point.py models/
```

Add a branch to `build_model` in `model_factory.py`:

```python
    elif model_type == "qwen3_5_point":
        from models.qwen3_5_point import Qwen3_5PointModel

        model = Qwen3_5PointModel()
        if args.model_name_or_path:
            model.load_model(model_name_or_path=model_name_or_path)
        else:
            model.load_model()
```

Then, on a GPU with a transformers version that supports Qwen3.5:

```bash
python eval_screenspot_pro.py \
  --model_type qwen3_5_point --model_name_or_path Qwen/Qwen3.5-4B \
  --screenspot_imgs ../data/screenspot_pro/images --screenspot_test ../data/screenspot_pro/annotations \
  --task all --language en --gt_type positive --log_path results/qwen3_5_point_base.json --inst_style instruction
```

## Evaluating a trained checkpoint

Merge the LoRA into a full HF model first (`python -m deskmind_eyes.export_ckpt` for a Tinker checkpoint,
`python -m deskmind_eyes.merge_hf_lora` for an `rl_hf` / `train_hf` adapter), then pass the merged directory to
`--model_name_or_path`.
