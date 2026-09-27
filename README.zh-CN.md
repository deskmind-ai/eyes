<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/brand/banner-dark.svg">
    <img src="assets/brand/banner-light.svg" alt="得心 DeskMind — 得心，应手。" width="720">
  </picture>
</p>

<p align="center">
  <b>DeskMind Eyes · 得心</b>：输入截图，输出点位。一个小型 GUI 定位模型，以及训练它的完整配方。<br>
  <a href="README.md">English</a> · <a href="docs/results.zh-CN.md">成绩</a> · <a href="docs/training.zh-CN.md">训练</a>
</p>

---

**得心，应手。** 「得心」（DeskMind）取自「得心应手」：心里想到，手上就做到。它是一组开源项目，让 agent 在你的电脑上**看**懂屏幕、**想**好下一步、**做**到真实的桌面上，全程在本机完成。

本仓库是其中的 **Eyes（视觉定位）**：给它一张截图和一句元素描述（比如「reload CMake cache」「发送按钮」），它返回该点哪里。仓库包括：
- 训练代码：SFT，以及用「点在框内得 1 分」做奖励的 GRPO 式强化学习；
- ScreenSpot v1 / v2 / Pro 和 UI-Vision 的评测工具；
- 一个本地定位服务；
- 我们报告的每次评测的汇总表。

| | 仓库 | 分工 |
|---|---|---|
| 👁 | **deskmind-ai/eyes** | 在截图里找到要操作的目标 |
| 🧠 | [deskmind-ai/brain](https://github.com/deskmind-ai/brain) | 决定下一步，并给出把握有多大 |
| ✋ | [deskmind-ai/hands](https://github.com/deskmind-ai/hands) | 在真实的 macOS 桌面上执行 |
| 📐 | [deskmind-ai/bench](https://github.com/deskmind-ai/bench) | 沙箱桌面任务和评分器，用来复现我们的成绩 |

## 特点

- **一个问题，一个点。** 输入截图和指令，输出 0–1000 相对坐标的点：`{"point_2d": [x, y]}`，或一次 `computer_use` 工具调用。点落在目标框内算对。
- **小模型，在题库上做 RL。** Eyes-4B = GUI-Owl-1.5-4B-Instruct + 用 RL 训练的 LoRA。每一轮有效的提升都来自同一件事：喂给模型它「时对时错」的题（见 [docs/training.zh-CN.md](docs/training.zh-CN.md)）。
- **本地运行。** `deskmind_eyes.ground_server` 在 Apple Silicon 上用 MLX 运行模型，只监听 `127.0.0.1`。请求里写的是本机 PNG 的路径，截图不离开你的电脑。
- **评测讲究。** 每组对比都在同一硬件、同一推理栈上逐条配对，并给出 McNemar z 值。各次评测的汇总在 [`results/`](results)。

## 成绩（2026 年 9 月）

ScreenSpot-Pro 全集（1,581 条）。所有数字都是我们自己在同一套环境里跑的：单张 GPU、vLLM 0.19、bf16、greedy 解码、原始分辨率、Qwen3-VL `computer_use` 工具调用 prompt。

| | 不放大（单次前向） | 放大 0.5（两次前向） |
|---|---|---|
| GUI-Owl-1.5-4B-Instruct（基座） | 64.8 | 76.2 |
| KV-Ground-4B，同环境复测 | 66.1 | – |
| **Eyes-4B** | **67.7** | **77.5** |

- **不放大**是主指标：对整张截图只做一次前向。
  - 对比基座：+2.9（多对 81 条 / 多错 34 条，z = 4.4）。
  - 对比同环境下的 KV-Ground-4B：+1.6（+68 / −42，z = 2.5）。
- **放大 0.5** 是第二次前向：以第一次的预测为中心，裁出宽高各一半的区域，再预测一次。对比基座 +1.3（z = 2.0），刚好在我们「显著」标准的边上。
- **其他测试集：**
  - ScreenSpot-v2（1,272 条）：92.8 → 95.0（z = 4.7）。mobile / desktop / web 的 text 和 icon 六项全部上升。
  - UI-Vision 元素定位（5,479 条）：30.7 → 32.5（z = 5.3），这是没用 GroundCUA 训练的 checkpoint。最终模型是 33.1，但 GroundCUA 和 UI-Vision 取自同一批 app，所以这个数不算干净。

**和别人比。** 其他模型公布的数字用的是各自的评测环境：
- 自报（不放大）：KV-Ground-4B 67.0，GUI-Owl-1.5-4B 66.8，Qwen-UI-Agent-4B 67.8（未开放权重）；
- KV-Ground-4B 在我们的环境里比它自报的低 0.9；
- 同一份权重换 GPU、换推理栈，Pro 上能差 0.5–0.9。

所以我们只声称配对实验能支持的结论：在同一环境下，Eyes-4B 超过了我们能跑到的两个开放权重 4B 定位模型；和 Qwen-UI-Agent-4B 自报的 67.8 相比，差距在噪声以内。

**已知不足：**
- **小目标：** 按框面积最小的那 1/4 目标，只找对了 46.1%。KV-Ground 在这部分和我们持平（46.6），在 Creative 类 icon 上领先（46.9 vs 44.8）。
- **放大：** 两次前向相对基座的提升不大（+1.3）。
- **部署配置：** 本地服务跑的是 4-bit MLX 版本，图片不超过 2M 像素，用 `point_2d` prompt。这不是评测时的配置，我们没有这套配置的成绩。

方法、分项成绩和早期 Qwen3.5-4B 的实验见 [docs/results.zh-CN.md](docs/results.zh-CN.md)。

## 快速上手

```bash
uv sync --extra mlx                     # Python 3.12；mlx 扩展用于 Apple Silicon
uv run python -m deskmind_eyes.prepare_data --only screenspot_pro     # 下载并逐张校验图片
```

**本地定位服务**（Apple Silicon）。Eyes-4B 权重暂未公开。可以按 [docs/training.zh-CN.md](docs/training.zh-CN.md) 自己训练得到，也可以先把 `--model` 指向 GUI-Owl-1.5-4B-Instruct 的任意 MLX 转换版来试用：

```bash
uv run python -m deskmind_eyes.ground_server --model models/eyes-4b-mlx --port 8010
curl -s localhost:8010/ground -H 'Content-Type: application/json' \
  -d '{"image": "/abs/path/to/screenshot.png", "queries": {"send": "the send message button"}}'
# -> {"points": {"send": [0.97, 0.95]}, "seconds": ...}      相对图片的 0-1 坐标；解析失败为 null
```

**跑评测**：CUDA GPU + vLLM 0.19，模型为合并后的 HF checkpoint：

```bash
uv pip install "vllm==0.19.*"
uv run python -m deskmind_eyes.eval_vllm --model runs/merged/eyes-4b --benchmark screenspot_pro --prompt-style tool
uv run python -m deskmind_eyes.eval_vllm --model runs/merged/eyes-4b --benchmark screenspot_pro --prompt-style tool --zoom 0.5
uv run python -m deskmind_eyes.compare runs/eval/base.jsonl runs/eval/candidate.jsonl      # 逐条配对，McNemar
```

在 Mac 上，`python -m deskmind_eyes.eval_mlx MODEL OUT.jsonl --benchmark screenspot_v2` 用 mlx-vlm 跑同一套协议（Pro 原始分辨率会很慢）。`leaderboard/` 里是给 ScreenSpot-Pro 官方评测脚本用的适配器。

## 训练

见 [docs/training.zh-CN.md](docs/training.zh-CN.md)，内容包括：
- 数据来源：ShowUI-desktop、改写过指令的 OS-Atlas desktop、GroundCUA；
- 难度打分和动态采样；
- 从 64.8 到 67.7 的三轮 RL；
- 试过但没用的做法。

## 品牌

方框是你的桌面，橙色的点是目光落下的地方。吉祥物叫**小方**，是那个方框活了过来。素材在 [`assets/brand`](assets/brand)。

<p>
  <img src="assets/brand/eyes-lockup-light.svg" height="48" alt="DeskMind Eyes">
</p>

<p>
  <img src="assets/brand/xiaofang-idle.svg" height="72" alt="小方·静候">
  <img src="assets/brand/xiaofang-think.svg" height="72" alt="小方·思考">
  <img src="assets/brand/xiaofang-working.svg" height="72" alt="小方·执行">
  <img src="assets/brand/xiaofang-done.svg" height="72" alt="小方·完成">
</p>

## 许可

代码采用 Apache-2.0。数据集和基座模型都从各自的来源下载，遵循各自的许可，本仓库不做再分发。DeskMind、得心这两个名称，以及 logo 和小方，不在代码许可范围内：可以用来指代本项目，但不能修改后使用，也不能暗示我们为别的产品背书。
