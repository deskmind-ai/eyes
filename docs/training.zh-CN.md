# 训练 · [English](training.md)

Eyes-4B = GUI-Owl-1.5-4B-Instruct + 在单张 GPU 上用强化学习训练的 LoRA。本文是 [results.zh-CN.md](results.zh-CN.md) 中各项成绩背后的配方，以及早期在 Qwen3.5-4B 上积累的经验。

## 任务和奖励

- **输入：** 一张截图和一句指令。
- **输出：** 一次点击，0–1000 相对坐标，格式是 Qwen3-VL 的 `computer_use` 工具调用（`--prompt-style tool`，即 GUI-Owl 的原生格式）。早期 Qwen3.5 阶段用的是 `{"point_2d": [x, y]}` prompt。
- **奖励：** 点落在目标框内得 1，否则得 0。
- **优势：** 奖励减去组内均值。组内奖励全部相同的组直接跳过。

## RL 设置（`deskmind_eyes.rl_hf`）

- **LoRA：** rank 32，作用于语言模型的全部线性层和 `lm_head`，视觉塔冻结。
- **采样：** 每条题目采 8 个答案，temperature 1.0（不用 top-k / top-p），每步 32 条题目。
- **优化器：** AdamW(0.9, 0.95, eps 1e-8, 无 weight decay)，lr 4e-5，到最后一步线性衰减到 10%。
- **损失：** 每个答案 −Σ A · log p(token)。rollout 直接用 policy 自己的 HF `generate`，不需要同步权重。
- **动态采样**（`--dynamic-sampling`，DAPO 式）：每步不断抽题，最多抽 4 × 32 条，直到凑满 32 个「有对有错」的组。

## 数据

所有数据都在运行时从原始来源下载，本仓库不做再分发。

| 数据池 | 来源 | 处理 |
|---|---|---|
| `showui_desktop` | showlab/ShowUI-desktop | 原样使用（`prepare_data`） |
| `osatlas_full_v3` | OS-Atlas desktop（Windows / Linux / macOS） | 只用全分辨率截图；用前沿模型把简短的无障碍名称改写成 ScreenSpot-Pro 风格的功能性指令（改写工具不在本仓库） |
| `groundcua` | ServiceNow/GroundCUA | 功能性指令（`prepare_groundcua`） |

- **为什么要改写 OS-Atlas 的指令。** OS-Atlas 原始名称（如 "Reload"，中位数只有 2 个词）教会模型逐字匹配文字，而 ScreenSpot-Pro 问的是意图（如 "reload CMake cache"）。
  - 给前沿模型看整张带编号框的截图，外加每个元素的放大裁剪图，让它为每个框写一句指令。改写工具不在本仓库里。
  - **说明：** 改写用的提示词里，拿了几条 ScreenSpot-Pro 测试集的指令（仅文字）当风格示例，没有用截图、标注框或答案。改写后的数据只用于训练，从未用于评测。
- **数据同源提醒。** GroundCUA 和 UI-Vision 取自同一批 app。`dedup_uivision` 可以找出两者之间近似重复的截图。最终模型训练时没有做这一步过滤，所以它的 UI-Vision 成绩不干净。

## 难度池

一组 8 个答案如果全对或全错，就没有梯度。所以先用模型自己给每条题目打通过率，只拿它「时对时错」的题来训练：

```bash
python -m deskmind_eyes.difficulty_vllm --model models/GUI-Owl-1.5-4B-Instruct --prompt-style tool \
    --data showui_desktop,osatlas_full_v3 --data-dir data/showui_desktop --osatlas-dir data/osatlas_desktop \
    --out runs/difficulty/guiowl_tool.jsonl
```

`rl_hf --difficulty FILE --min-pass A --max-pass B` 只保留通过率在 [A, B] 之间的题。

## 三轮训练

```bash
M=models/GUI-Owl-1.5-4B-Instruct
DATA="--data showui_desktop,osatlas_full_v3 --data-dir data/showui_desktop --osatlas-dir data/osatlas_desktop"

# 1. 在难度池上做 GRPO（通过率 0.1-0.9，约 5.4K 条）：Pro 64.8 -> 65.8
python -m deskmind_eyes.rl_hf --model $M --prompt-style tool --max-tokens 64 $DATA \
    --difficulty runs/difficulty/guiowl_tool.jsonl --max-steps 40 --out runs/eyes-grpo

# 2. 同一个池，改用动态采样，同样从基座开始：66.7
python -m deskmind_eyes.rl_hf --model $M --prompt-style tool --max-tokens 64 $DATA \
    --difficulty runs/difficulty/guiowl_tool.jsonl --dynamic-sampling --max-steps 40 --out runs/eyes-dapo

# 3. 接着第 2 轮，换更难、更新的 GroundCUA 池，2.5M 像素（通过率 1/8-4/8，约 3.3K 条）：67.7
python -m deskmind_eyes.difficulty_vllm --model $M --prompt-style tool --data groundcua --max-pixels 2500000 \
    --out runs/difficulty/groundcua_tool_2p5m.jsonl
python -m deskmind_eyes.rl_hf --model $M --prompt-style tool --max-tokens 64 --data groundcua \
    --difficulty runs/difficulty/groundcua_tool_2p5m.jsonl --min-pass 0.1 --max-pass 0.5 --max-pixels 2500000 \
    --init-adapter runs/eyes-dapo/final --dynamic-sampling --max-steps 40 --save-every 10 --out runs/eyes-groundcua
```

**第三轮为什么收紧池子。** 第 2 轮之后，在 [1/8, 7/8] 的 GroundCUA 池上，第 0 步就有 61% 的组全对。只保留基座最多答对一半的题之后，「有对有错」的组又回来了。

**合并、评测、部署：**

```bash
python -m deskmind_eyes.merge_hf_lora --base $M --adapter runs/eyes-groundcua/final --out runs/merged/eyes-4b
python -m deskmind_eyes.eval_vllm --model runs/merged/eyes-4b --benchmark screenspot_pro --prompt-style tool
python -m mlx_vlm.convert --hf-path runs/merged/eyes-4b --mlx-path models/eyes-4b-mlx -q --q-bits 4   # 在 Mac 上
```

`merge_hf_lora` 可以传多个 `--adapter` 目录做权重平均；在我们的实验里最多 +0.1。

## 早期阶段：在 Tinker 上训练 Qwen3.5-4B

`deskmind_eyes.train`（SFT）和 `deskmind_eyes.rl` 运行在 [Tinker](https://thinkingmachines.ai/tinker/) 上，需要在环境变量里设置 `TINKER_API_KEY`。它们和 `rl_hf` 是同一套算法；`train_hf` 是对应的单 GPU SFT 版本。这些训练的配置和逐步指标在 `results/training/`。

```bash
python -m deskmind_eyes.eval_tinker benchmark=screenspot_pro native=True
python -m deskmind_eyes.difficulty sources=showui_desktop out=runs/difficulty/showui_desktop.jsonl
python -m deskmind_eyes.rl difficulty_file=runs/difficulty/showui_desktop.jsonl min_pass_rate=0.1 max_pass_rate=0.9 \
    eval_benchmarks=screenspot_v2,screenspot_pro eval_native=True max_steps=20 eval_every=10
```

## 每条都花掉过一轮训练的教训

- **基座最重要。** 换一次基座的收益，超过在旧基座上的全部训练。
- **RL 的瓶颈在题库，不在算法。** 每一次提升都来自给模型「它时对时错的题」：难度过滤、动态采样、用当前 policy 重新打分、换新的数据源。静态过滤很快会过时：在 Qwen3.5 上，40 步之内有梯度的组从 45% 降到了 6%。
- **检查图片是怎么送进模型的。** 最早的评测在发给采样服务的路上把截图压成了 JPEG，ScreenSpot-Pro 因此被低估了 5 分，训练图片也经过了同样的压缩。
- **提升可能和输出格式绑定。** 用 `point_2d` prompt 训练出的 RL 提升，换成工具调用 prompt 后只剩三分之一。用什么格式部署，就用什么格式训练。
- **训练分辨率更高不一定更好。** 在 Qwen3.5 上，用原始分辨率代替约 1M 像素训练并没有更好（Pro 55.5 vs 56.9，不显著）：更多组 8 个答案全对，难题贡献的梯度反而更少。
- **只在同一平台内比较。** 同一个 checkpoint 换 GPU 和推理栈，Pro 上最多差 0.9，和某些训练带来的提升一样大。
