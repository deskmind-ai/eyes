# 成绩 · [English](results.md)

所有数字都是我们自己跑的。[`results/eval_summary.md`](../results/eval_summary.md) 和 [`results/eval_summary.csv`](../results/eval_summary.csv) 由逐条评测输出生成，覆盖本文引用的每一次评测。

## 评测协议

- **环境。** 单张 GPU、vLLM 0.19、bf16、greedy 解码、截图保持原始分辨率（`eval_vllm`）。
- **Prompt。** Qwen3-VL `computer_use` 工具调用 prompt（`--prompt-style tool`），答案是一次带 0–1000 坐标的 `left_click`。
- **判分。** 点落在目标框内算对。
- **放大 0.5**（`--zoom 0.5`）是第二次前向，也就是 KV-Ground 和 GUI-Owl 所说的 zoom：以第一次的预测为中心，裁出宽高各一半的区域（碰到边界就截断），放大回原图尺寸，再预测一次，然后把点映射回原图。
- **配对检验。** 每组对比都在同一批题目上逐条配对。`+` / `−` 是只有一方答对的题数；z = (+ − −) / √(+ + −)，即 McNemar 统计量。|z| ≥ 2 视为显著。
- **平台差异是真实存在的。** 同一份权重换一种 GPU 和推理栈，ScreenSpot-Pro 能差 0.5–0.9；分辨率较低的 ScreenSpot-v2 上没有差异。所以我们只在同一环境内做比较。

## Eyes-4B

### ScreenSpot-Pro 全集（1,581 条）

| 不放大 | CAD text | CAD icon | Creative text | Creative icon | Dev text | Dev icon | OS text | OS icon | Office text | Office icon | Scientific text | Scientific icon | **平均** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GUI-Owl-1.5-4B 基座 | 59.4 | 39.1 | 72.7 | 41.3 | 82.5 | 51.0 | 81.3 | 49.4 | 85.9 | 49.1 | 85.4 | 41.8 | 64.8 |
| KV-Ground-4B | 57.4 | 39.1 | 77.3 | 46.9 | 81.8 | 49.0 | 74.8 | 51.7 | 88.1 | 54.7 | 88.2 | 47.3 | 66.1 |
| **Eyes-4B** | 62.4 | 43.8 | 77.3 | 44.8 | 84.4 | 51.7 | 80.4 | 51.7 | 89.3 | 56.6 | 88.2 | 46.4 | **67.7** |

| 放大 0.5 | CAD text | CAD icon | Creative text | Creative icon | Dev text | Dev icon | OS text | OS icon | Office text | Office icon | Scientific text | Scientific icon | **平均** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GUI-Owl-1.5-4B 基座 | 83.8 | 59.4 | 81.8 | 56.6 | 87.7 | 59.3 | 79.4 | 66.3 | 92.1 | 75.5 | 89.6 | 55.5 | 76.2 |
| **Eyes-4B** | 82.2 | 59.4 | 83.8 | 56.6 | 88.3 | 62.8 | 83.2 | 66.3 | 93.2 | 77.4 | 91.0 | 60.9 | **77.5** |

| 配对比较 | + | − | z |
|---|---|---|---|
| Eyes-4B 对比基座，不放大 | 81 | 34 | 4.38 |
| Eyes-4B 对比 KV-Ground-4B，不放大 | 68 | 42 | 2.48 |
| Eyes-4B 对比基座，放大 0.5 | 74 | 52 | 1.96 |

- **错题互补。** KV-Ground-4B 在 Creative 类 icon 上更强；我们在 text 类和较大的目标上更强。两者「任一答对」就算对的话，Pro 能到 70.4。这说明高分辨率下的小目标是补数据最有价值的方向。
- **其他模型在各自环境里的自报数字：**
  - KV-Ground-4B：67.0（我们环境里 66.1）；
  - GUI-Owl-1.5-4B：66.8；
  - Qwen-UI-Agent-4B：67.8（未开放权重，我们无法复测）。

### 按 Mac 应用的实际设置（4 位 MLX，≤200 万像素）

同样 1,581 条，用 DeskMind 应用实际使用的权重和设置：`deskmind/eyes-4b` 的 `mlx-4bit` 版本（de8e13b），应用自带运行时里的 mlx-vlm 0.7.4，单次前向，截图缩到不超过 2,000,000 像素（`ground_server` 的默认值），贪心解码，point 提示词。Apple M4 Pro，48 GB。

| | CAD text | CAD icon | Creative text | Creative icon | Dev text | Dev icon | OS text | OS icon | Office text | Office icon | Scientific text | Scientific icon | **平均** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Eyes-4B，GPU bf16，原始分辨率（见上） | 62.4 | 43.8 | 77.3 | 44.8 | 84.4 | 51.7 | 80.4 | 51.7 | 89.3 | 56.6 | 88.2 | 46.4 | 67.7 |
| **Eyes-4B，Mac 应用：4 位 MLX，≤200 万像素** | 38.6 | 23.4 | 64.6 | 21.7 | 74.0 | 26.2 | 65.4 | 30.3 | 75.1 | 41.5 | 77.8 | 34.5 | **50.9** |

- 文字目标 64.8，图标目标 28.3；全部输出都能解析。
- 掉分主要在图标（每组低 20–28 分）。ScreenSpot-Pro 的截图大多是 4K 级别，缩到 200 万像素约缩小四倍，小图标损失最大。分辨率和量化各占多少，正在测。
- 每次定位耗时：中位数 4.7 秒，p95 7.8 秒（约 2,000 个提示词 token）；MLX 内存峰值 4.5 GB。测评期间部分时间和另一个模型服务同时运行，耗时按上限看。
- 命令：`python -m deskmind_eyes.eval_mlx <model> out.jsonl --benchmark screenspot_pro --max-pixels 2000000`。

#### 分辨率：2 MP 与 4 MP

差距里有多少来自分辨率：同一份 4 位权重，在 300 条子集（seed 0）上分别用 2 MP（应用的设置）和 4 MP 测。

| 同样 300 条 | 合计 | 文字 | 图标 | 每次中位耗时 | p95 | 提示词 token | 内存峰值 |
|---|---|---|---|---|---|---|---|
| 2 MP | 49.3 | 63.1 | 23.8 | 4.7 秒 | 7.9 秒 | 约 2,000 | 4.5 GB |
| 4 MP | **59.0** | 73.8 | 31.4 | 11.9 秒 | 17.2 秒 | 约 4,000 | 4.9 GB |

- +9.7 分（多对 42 条、多错 13 条，z = 3.9）：和 GPU 成绩的差距大部分来自分辨率，其余来自 4 位量化，以及原始截图比 4 MP 更大。
- 每次定位慢约 2.5 倍。两次测量都有部分时间和另一个模型服务同时运行，绝对耗时按上限看，两者的比例是可信的。
- 决定：应用 0.3.1 保持 2 MP，因为每个视觉步骤多等约 7 秒，用户能明显感觉到。下一步在 Mac 上做两遍定位：先用 2 MP 粗定位，再裁剪局部看一次（GPU 上放大 0.5 得 77.5）。
- 命令：在上面的命令后加 `--n 300`，再分别用 `--max-pixels 2000000` 或 `4000000`。

### 最小的目标（Pro，不放大）

| | 框最小的四分之一目标 | Creative icon |
|---|---|---|
| KV-Ground-4B | 46.6 | 46.9 |
| **Eyes-4B** | 46.1 | 44.8 |

小目标是 Eyes-4B 已知的短板：在框最小的四分之一目标上 KV-Ground 与之持平，在 Creative 类应用的 icon 上领先。

### ScreenSpot-v2（1,272 条）和 UI-Vision（5,479 条）

| ScreenSpot-v2 | mobile text | mobile icon | desktop text | desktop icon | web text | web icon | **平均** |
|---|---|---|---|---|---|---|---|
| GUI-Owl-1.5-4B 基座 | 98.3 | 89.1 | 95.4 | 85.0 | 95.7 | 88.7 | 92.8 |
| **Eyes-4B** | 99.3 | 91.5 | 97.4 | 88.6 | 97.0 | 92.1 | **95.0** |

配对：+30 / −3，z = 4.7。

| UI-Vision 元素定位 | basic text | basic icon | functional text | functional icon | spatial text | spatial icon | **平均** |
|---|---|---|---|---|---|---|---|
| GUI-Owl-1.5-4B 基座 | 66.9 | 32.7 | 58.5 | 31.2 | 45.9 | 12.6 | 30.7 |
| Eyes-4B-DAPO（未用 GroundCUA） | 72.0 | 33.8 | 64.2 | 33.7 | 45.9 | 13.1 | **32.5** |
| Eyes-4B（最终版）¹ | 73.9 | 34.6 | 68.4 | 33.8 | 47.3 | 12.8 | 33.1 |

配对，DAPO 对比基座：+228 / −128，z = 5.3。

¹ 这个数不干净。UI-Vision 的所有 app 都出现在 GroundCUA 里，而最后一轮是在 GroundCUA 上训练的；GroundCUA 论文也没有提到对截图做过去重。ScreenSpot-Pro（商业软件）和 ScreenSpot-v2 不受影响。`deskmind_eyes.dedup_uivision` 可以找出两个数据集之间近似重复的截图。

### 从 64.8 到 67.7（Pro，不放大）

| 步骤 | Pro | 增量 |
|---|---|---|
| GUI-Owl-1.5-4B-Instruct，不训练 | 64.8 | |
| GRPO，40 步，5.4K 条难度池 | 65.8 | +1.0 |
| 改用 DAPO 式动态采样，40 步，同一个池 | 66.7 | +0.9 |
| 在 GroundCUA 池上继续动态采样，2.5M 像素，40 步 | **67.7** | +1.0 |

GRPO 和 DAPO 两轮都从基座开始；GroundCUA 这一轮接着 DAPO 的 adapter 训练。细节见 [training.zh-CN.md](training.zh-CN.md)。

**试过但没有提升的：**
- **checkpoint 权重平均：** DAPO 66.9 vs 66.7，GroundCUA 67.6 vs 67.7。
- **GUI-Owl 官方评测 prompt**（官方 system prompt，图片上限 9,800 个视觉 token）：在我们的推理栈上基座反而降到 63.2，复现不出 GUI-Owl 自报的 66.8。

## 早期阶段：在 Tinker 上训练 Qwen3.5-4B

换基座之前，我们在 [Tinker](https://thinkingmachines.ai/tinker/) 上对 Qwen3.5-4B 做了 SFT 和 RL。这些数字用的是我们的 `point_2d` prompt 和 Tinker 采样，只适合相互比较。

| Pro，原始分辨率 | JPEG 传图 | 无损传图 |
|---|---|---|
| Qwen3.5-4B 基座 | 52.9 | 58.1 |
| RL，难度过滤池，第 20 步 | 59.8 | 61.9 |
| RL，动态采样，第 30 步 | 59.6 | 63.2 |

- **传图方式有影响。** 最早的评测把截图压成质量 75 的 JPEG 再发送，基座在 Pro 上因此少了 5 分。`deskmind_eyes.image_encoding` 改为发送 PNG 或 4:4:4 高质量 JPEG。ScreenSpot-v2 几乎不受影响（91.3 → 91.1）。
- **ScreenSpot-v2，JPEG 传图：** 基座 91.3；难度过滤 RL 93.5；动态采样 94.0。
- **提升和输出格式绑定。** 换成社区版 ScreenSpot-Pro 适配器（工具调用 prompt）后，难度过滤 RL 相对基座只涨 +2.2（49.0 → 51.1）；在训练所用的 prompt 下是 +6.9。
- **基座最重要。** Qwen3.5-4B 最好的一次（63.2，Tinker）仍低于不训练的 GUI-Owl-1.5-4B（64.8，vLLM）。评测环境不同，但换基座显然是单步最大的提升。

## 文件

- `results/eval_summary.md`、`results/eval_summary.csv`：每一次评测，按组列出分项准确率，以及上文的配对比较。
- `results/training/<run>/`：Qwen3.5-4B 在 Tinker 上各次训练的配置和逐步指标（`metrics.csv`）。
