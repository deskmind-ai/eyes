"""Official ScreenSpot-Pro evaluation protocol for Qwen3.5, vendored verbatim.

Source: github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding models/qwen3_5.py
(HF path `Qwen3_5Model.ground_only_positive`; the vLLM path pre-fills GUIDED_PREFILL).
"""

import json
import math

# System prompt text exactly as the official adapter sends it.
SYSTEM_TEXT = 'You are a helpful assistant. The user will give you an instruction, and you MUST left click on the corresponding UI element via tool call. If you are not sure about where to click, guess a most likely one.\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{"type": "function", "function": {"name": "computer_use", "description": "Use a mouse to interact with a computer.\n* The screen\'s resolution is 1000x1000.\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. \n* You can only use the left_click action to interact with the computer.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\n* `left_click`: Click the left mouse button with coordinate (x, y).", "enum": ["left_click"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=left_click`.", "type": "array"}, "required": ["action"], "type": "object"}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>'

# vLLM path: the assistant turn is pre-filled with the start of the tool call.
GUIDED_PREFILL = '<tool_call>\n{"name": "computer_use", "arguments": {"action": "left_click", "coordinate": ['

MAX_PIXELS = 8294400  # QWEN3_5_MAX_PIXELS default in the HF path (vLLM path: 2073600)
MIN_PIXELS = 32 * 32
RESIZE_FACTOR = 8  # HF path uses factor=8 before handing the image to the processor


def smart_resize(height: int, width: int, factor: int = RESIZE_FACTOR,
                 min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> tuple[int, int]:
    """Copy of transformers.models.qwen2_vl.image_processing_qwen2_vl.smart_resize."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def parse_response(response: str) -> tuple[float, float] | None:
    """Official parsing: JSON inside the last <tool_call> block; 2 coords = point, 4 = box center;
    coordinates are 0-1000 relative. Returns a relative (0-1) point or None."""
    try:
        action = json.loads(response.split("<tool_call>\n")[-1].split("\n</tool_call>")[-2])
        coordinates = action["arguments"]["coordinate"]
        if len(coordinates) == 2:
            point_x, point_y = coordinates
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            point_x, point_y = (x1 + x2) / 2, (y1 + y2) / 2
        else:
            raise ValueError("Wrong output format")
        return point_x / 1000, point_y / 1000
    except (IndexError, KeyError, TypeError, ValueError):
        return None
