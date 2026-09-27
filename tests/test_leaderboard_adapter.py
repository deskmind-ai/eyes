"""The leaderboard adapter must implement exactly the protocol our Tinker / MLX evals use."""

import importlib.util
from pathlib import Path

from PIL import Image

from deskmind_eyes import common

ADAPTER = Path(__file__).resolve().parent.parent / "leaderboard" / "models" / "qwen3_5_point.py"
spec = importlib.util.spec_from_file_location("qwen3_5_point", ADAPTER)
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_prompt_matches():
    assert adapter.PROMPT == common.PROMPT_STYLES["point"] == common.PROMPT


def test_resize_matches():
    assert adapter.MAX_PIXELS == common.NATIVE_MAX_PIXELS
    for size in [(1920, 1080), (3840, 2160), (5120, 4000), (6000, 3500)]:
        img = Image.new("RGB", size)
        assert adapter.preprocess_image(img).size == common.resize_to_max_pixels(img, common.NATIVE_MAX_PIXELS).size


def test_parse_matches():
    cases = [
        '```json\n[\n\t{"point_2d": [412, 87], "label": "x"}\n]\n```',
        '{"point_2d": [0.4, 0.9]}',
        "<think>\nsee [1, 2]\n</think>\n\n{\"point_2d\": [640, 360]}",
        "no coordinates here",
        '{"point_2d": [12.5, 999]}',
    ]
    for text in cases:
        assert adapter.parse_response(text) == common.parse_point(text)


if __name__ == "__main__":
    test_prompt_matches()
    test_resize_matches()
    test_parse_matches()
    print("adapter matches deskmind_eyes.common")
