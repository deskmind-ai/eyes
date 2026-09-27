# 为得心 Eyes 做贡献 · [English](CONTRIBUTING.md)

感谢参与！得心 Eyes 是个小项目，所有改动都用同一个标准衡量：在别人也能复现的对比里，它是不是更常把点落在正确的元素上？组织层面的通用规范（[deskmind-ai/.github](https://github.com/deskmind-ai/.github)）在这里同样适用，包括[行为准则](https://github.com/deskmind-ai/.github/blob/main/CODE_OF_CONDUCT.md)。

## 环境

```bash
git clone https://github.com/deskmind-ai/eyes && cd eyes
uv sync --extra test             # Python 3.12；Apple Silicon 上再加 --extra mlx
uv run pytest -q                 # 提 PR 前必须通过
```

- 测试不需要 GPU、模型或数据集，只检查排行榜适配器的提示词、缩放和解析与我们的实现完全一致（`tests/test_leaderboard_adapter.py`）。
- 评测和训练需要装有 vLLM 0.19 的 CUDA GPU（`uv pip install "vllm==0.19.*"`）。MLX 路径（`eval_mlx`、`ground_server`）在 Apple Silicon Mac 上运行。

## 复现评测

```bash
uv run python -m deskmind_eyes.prepare_data --only screenspot_pro        # 从数据源下载并校验图片
uv run python -m deskmind_eyes.eval_vllm --model MODEL --benchmark screenspot_pro --prompt-style tool --out runs/eval/a.jsonl
uv run python -m deskmind_eyes.compare runs/eval/base.jsonl runs/eval/a.jsonl   # 配对比较，McNemar z
```

- **只在同一套环境内比较。** 同一份权重换 GPU 或推理框架，在 ScreenSpot-Pro 上会差 0.5–0.9 分。基线请在你自己的机器上、用相同的提示风格和分辨率自己跑。
- **报告配对结果：** 多对了几个、多错了几个、z 值。|z| ≥ 2 我们才称为显著，更小的只报数字、不下结论。
- 逐样本输出（`runs/eval/*.jsonl`）不提交，`results/` 里只放汇总表。

## 复现训练

[docs/training.zh-CN.md](docs/training.zh-CN.md) 列出了三轮 RL 的全部命令，从难度打分到合并和部署，每一轮都只需一张 GPU。有两处容易出错：

- **图片传输。** 截图必须无损地送到模型。`deskmind_eyes/image_encoding.py` 就是因为采样时的 JPEG 压缩让 Pro 掉了 5 分才加的。
- **模型生成的训练数据。** 如果 PR 加入了由模型写的指令或标注，请在 PR 里写明是哪个模型写的、依据什么条款使用。

训练类 PR 要写明基座模型、数据池及其规模、步数、硬件，以及同一环境下训练前后的评测结果。

## 新增评测集

1. **下载。** 在 `deskmind_eyes/prepare_data.py` 的 `SOURCES` 里加一项：Hugging Face 数据集仓库、本地目录、元数据文件、如何列出图片（图片打成一个 zip 发布时用 `archive`）。
2. **加载。** 把名字加进 `deskmind_eyes/common.py` 的 `BENCHMARKS`，并在 `load_benchmark` 里加一个分支，按统一格式返回样本：`img_path`、`img_filename`、`instruction`、`bbox_xyxy`（像素）、`group`、`data_type`。`eval_vllm`、`eval_mlx` 和 `compare` 会自动支持。
3. **核对。** 用基座模型跑一遍，把样本数和基座准确率与数据集作者公布的数字对比，都写进 PR。
4. **写文档。** 在 [docs/results.md](docs/results.md) 及其中文版里为这个评测集加一节，写明数据集许可和来源链接。

## 数据集许可

- **数据集在运行时下载，从不再分发。** 不要提交图片、标注或由它们派生的任何文件（裁剪图、改写后的指令、难度分），除非数据集许可允许。`data/`、`runs/`、`models/` 不提交到 git 正是为此。
- **遵守每个许可。** 新增数据源前先读它的许可和使用条款，并在 PR 里写明。许可禁止这种用途或没有许可的数据集不予添加。
- **说明可能的污染。** 如果训练数据与某个评测集共享应用或截图（例如 GroundCUA 和 UI-Vision），在对应数字旁边写明。
- 基座模型和权重同理：给链接，不要在这里重新上传。

## 提交 PR

- **一个 PR 只做一件事**，简单说明改了什么、怎么验证的。
- **准确率的结论要有配对证据。** 贴出在同一环境下与基线对比的 `compare` 输出，并写明评测集、提示风格、分辨率和硬件。
- **保持输出格式。** `ground_server` 返回 `{"points": {name: [x, y] | null}}`，坐标是 0–1 相对值，DeskMind Hands 依赖这个格式；模型输出是 0–1000 相对坐标。扩展时两者都要保持兼容。
- **代码风格：** 跟周围代码保持一致，函数小，注释写「为什么」，不引入新框架。
- **检查清单：**
  - [ ] `uv run pytest -q` 通过；
  - [ ] diff 里没有数据集文件、逐样本输出、权重、你自己应用的截图或密钥；
  - [ ] 新数据源写明了许可；
  - [ ] 中英文档同步更新（`docs/*.md` 和 `docs/*.zh-CN.md`）。

## 反馈问题

- **Bug：** 附上命令、模型、评测集和硬件，以及几个能说明问题的样本 id。
- **安全问题：** 不要公开提 issue，见 [SECURITY.md](https://github.com/deskmind-ai/.github/blob/main/SECURITY.md)。

提交贡献即表示你同意贡献内容按本仓库的 Apache-2.0 许可发布。数据集和基座模型保留各自的许可。
