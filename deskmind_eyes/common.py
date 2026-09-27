"""Shared prompt / parsing / preprocessing for GUI grounding (train + eval must match)."""

import json
import re
from collections import defaultdict
from pathlib import Path

from PIL import Image

try:  # Tinker-side code only; the MLX / leaderboard paths don't have tinker installed
    from deskmind_eyes.image_encoding import use_lossless

    IMAGE_FORMAT = use_lossless()
except ImportError:
    IMAGE_FORMAT = None

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SCREENSPOT_DIR = DATA_DIR / "screenspot"
SCREENSPOT_V2_DIR = DATA_DIR / "screenspot_v2"
SCREENSPOT_PRO_DIR = DATA_DIR / "screenspot_pro"
UI_VISION_DIR = DATA_DIR / "ui_vision"
SHOWUI_DESKTOP_DIR = DATA_DIR / "showui_desktop"

# ~1M pixels -> ~1000 image tokens for Qwen3.5 (32px per token after 2x2 merge).
# Used for training and the original ScreenSpot eval.
MAX_PIXELS = 1_003_520
# Qwen3.5 image processor's own ceiling (longest_edge = 16777216 pixels). "Native" eval
# only downsizes above this, so V2 / Pro screenshots keep their full resolution.
NATIVE_MAX_PIXELS = 16_777_216

PROMPT = (
    "Locate the UI element in the screenshot that matches the description, and "
    'output its click point in JSON as {{"point_2d": [x, y]}}, with coordinates '
    "in 0-1000 relative to the image.\nDescription: {instruction}"
)

_POINT_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)")
_NUM = r"(-?\d+(?:\.\d+)?)"
_BOX_RE = re.compile(rf"\[\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*,\s*{_NUM}\s*\]")

# Prompt styles for evaluation. "point" is what SFT/RL train on; the others exist to find
# the format the base model grounds best with (Qwen does not publish its eval prompt).
PROMPT_STYLES = {
    "point": PROMPT,
    "bbox": (
        "Locate the UI element in the screenshot that matches the description, and "
        'output its bounding box in JSON as {{"bbox_2d": [x1, y1, x2, y2]}}, with coordinates '
        "in 0-1000 relative to the image.\nDescription: {instruction}"
    ),
    "short": 'Point to "{instruction}" in the screenshot, output its coordinates in JSON format.',
    "tool": "{instruction}",
}

# Qwen3-VL's computer_use grounding prompt as used by KV-Ground / ScreenSpot-Pro-GUI-Grounding for Qwen3-VL and
# GUI-Owl-1.5 (github.com/vocaela/kv-ground utils.py): system message with the tool, the bare instruction as user text,
# answer `<tool_call>{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": [x, y]}}</tool_call>`
# in 0-1000. parse_response takes the first [x, y], which is the coordinate.
TOOL_SYSTEM = """
You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen's resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \n* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
""".strip()

# GUI-Owl-1.5's own grounding eval prompt (X-PLUG/MobileAgent, Mobile-Agent-v3.5/grounding_and_kb/eval_grounding_benchmarks.py,
# only_two_action_system_prompt): tools only, mouse_move + left_click, "terminate" when the element can't be found; used
# with images capped at 9800 visual tokens (MAX_PIXELS_OWL). GUI-Owl reports Pro 66.8 with it vs 65.3 under TOOL_SYSTEM.
OWL_SYSTEM = '# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen\'s resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don\'t click boxes on their edges unless asked.\n* don\'t use any other computer use tool like type, key, scroll, left_click_drag and so on.\n* you can only use the left_click and mouse_move action to interact with the computer. if you can\'t find the element, you should terminate the task and report the failure.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\n* `left_click`: Click the left mouse button with coordinate (x, y) pixel coordinate on the screen.", "enum": ["mouse_move", "left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=mouse_move` and `action=left_click`.", "type": "array"}}, "required": ["action"], "type": "object"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\n'
MAX_PIXELS_OWL = 9800 * 32 * 32

def grounding_messages(instruction: str, style: str = "point", image_part: dict | None = None) -> list[dict]:
    """Chat messages for one grounding query. point: our point_2d prompt (user turn only); tool: TOOL_SYSTEM + the bare
    instruction, answered as a computer_use tool call. parse_point reads either answer (first [x, y])."""
    image_part = image_part or {"type": "image"}
    if style in ("tool", "owl"):
        system = TOOL_SYSTEM if style == "tool" else OWL_SYSTEM
        return [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": [image_part, {"type": "text", "text": instruction}]}]
    return [{"role": "user", "content": [image_part, {"type": "text", "text": PROMPT.format(instruction=instruction)}]}]


COMPUTER_USE_TOOL = {
    "name": "computer_use",
    "description": (
        "Use a mouse to interact with the computer screen shown in the screenshot.\n"
        "* The screen's resolution is 1000x1000: coordinates are relative, (0, 0) is the "
        "top-left corner and (1000, 1000) the bottom-right corner.\n"
        "* Click on the center of the target element."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["left_click"], "description": "The action to perform."},
            "coordinate": {"type": "array", "description": "(x, y) in 0-1000 relative coordinates."},
        },
        "required": ["action", "coordinate"],
    },
}


def format_answer(x_rel: float, y_rel: float) -> str:
    """Relative (0-1) point -> target text the model is trained to emit."""
    return json.dumps({"point_2d": [round(x_rel * 1000), round(y_rel * 1000)]})


def parse_point(text: str) -> tuple[float, float] | None:
    """Model output -> relative (0-1) point. Accepts 0-1000 ints or 0-1 floats."""
    m = _POINT_RE.search(text.split("</think>")[-1])
    if not m:
        return None
    x, y = float(m.group(1)), float(m.group(2))
    if x <= 1 and y <= 1:
        return x, y
    return x / 1000, y / 1000


def parse_point_style(text: str, style: str) -> tuple[float, float] | None:
    """Parse a relative (0-1) click point for a given prompt style (bbox -> box center)."""
    if style == "bbox":
        m = _BOX_RE.search(text.split("</think>")[-1])
        if m:
            x1, y1, x2, y2 = (float(v) for v in m.groups())
            scale = 1 if max(x1, y1, x2, y2) <= 1 else 1000
            return (x1 + x2) / 2 / scale, (y1 + y2) / 2 / scale
    return parse_point(text)


def resize_to_max_pixels(img: Image.Image, max_pixels: int = MAX_PIXELS) -> Image.Image:
    w, h = img.size
    if w * h <= max_pixels:
        return img
    s = (max_pixels / (w * h)) ** 0.5
    return img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)


BENCHMARKS = ["screenspot", "screenspot_v2", "screenspot_pro", "ui_vision"]


def load_benchmark(name: str) -> list[dict]:
    """Samples in one schema: img_path, img_filename, instruction, bbox_xyxy (pixels),
    group (device for ScreenSpot v1/v2, professional domain for Pro), data_type (text/icon)."""
    samples = []
    if name in ("screenspot", "screenspot_v2"):
        root = SCREENSPOT_DIR if name == "screenspot" else SCREENSPOT_V2_DIR
        for device in ["mobile", "desktop", "web"]:
            fname = f"{device}.json" if name == "screenspot" else f"screenspot_{device}_v2.json"
            for s in json.load(open(root / fname)):
                x, y, w, h = s["bbox"]
                samples.append({
                    "img_path": root / "images" / s["img_filename"], "img_filename": s["img_filename"],
                    "instruction": s["instruction"], "bbox_xyxy": [x, y, x + w, y + h],
                    "group": device, "data_type": s["data_type"],
                })
    elif name == "screenspot_pro":
        for ann in sorted((SCREENSPOT_PRO_DIR / "annotations").glob("*.json")):
            for s in json.load(open(ann)):
                samples.append({
                    "img_path": SCREENSPOT_PRO_DIR / "images" / s["img_filename"],
                    "img_filename": s["img_filename"], "instruction": s["instruction"],
                    "bbox_xyxy": s["bbox"], "group": s["group"], "data_type": s["ui_type"],
                })
    elif name == "ui_vision":  # element grounding (ServiceNow/ui-vision): group = basic / functional / spatial
        for ann in sorted((UI_VISION_DIR / "annotations" / "element_grounding").glob("*.json")):
            setting = ann.stem.removeprefix("element_grounding_")
            for s in json.load(open(ann)):
                samples.append({
                    "img_path": UI_VISION_DIR / "images" / s["image_path"], "img_filename": s["image_path"],
                    "instruction": s["prompt_to_evaluate"], "bbox_xyxy": s["bbox"], "group": setting,
                    "data_type": s["element_type"], "platform": s["platform"],
                })
    else:
        raise ValueError(f"unknown benchmark {name!r}; choose from {BENCHMARKS}")
    return samples


def load_screenspot() -> list[dict]:
    return load_benchmark("screenspot")


def point_in_bbox(sample: dict, point_rel: tuple[float, float] | None, img_size) -> bool:
    if point_rel is None:
        return False
    W, H = img_size
    px, py = point_rel[0] * W, point_rel[1] * H
    x1, y1, x2, y2 = sample["bbox_xyxy"]
    return x1 <= px <= x2 and y1 <= py <= y2


def summarize(rows: list[dict]) -> dict:
    """rows need group (or legacy device) / data_type / hit / point -> ScreenSpot-style table."""
    groups = defaultdict(list)
    for r in rows:
        groups[f"{r.get('group', r.get('device'))}-{r['data_type']}"].append(r["hit"])
    table = {k: round(100 * sum(v) / len(v), 1) for k, v in sorted(groups.items())}
    table["avg"] = round(100 * sum(r["hit"] for r in rows) / len(rows), 1)
    table["n"] = len(rows)
    table["parse_fail"] = sum(r["point"] is None for r in rows)
    return table
