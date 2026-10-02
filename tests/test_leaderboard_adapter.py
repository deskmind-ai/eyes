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


def test_parse_expected_points():
    """Expected values, not only agreement: both parsers computing the same wrong point passed the test above."""
    cases = {
        '{"point_2d": [1, 1]}': (0.001, 0.001),          # 0-1000 protocol: the top-left corner, not the bottom-right
        '{"point_2d": [0, 0]}': (0.0, 0.0),
        '{"point_2d": [1000, 1000]}': (1.0, 1.0),
        '{"point_2d": [500, 250]}': (0.5, 0.25),
        '{"point_2d": [1, 0]}': (0.001, 0.0),
        '{"point_2d": [0.5, 0.25]}': (0.5, 0.25),        # written as fractions
        '{"point_2d": [1.0, 0.0]}': (1.0, 0.0),
        '{"point_2d": [1001, 10]}': None,                # outside the image
        '{"point_2d": [-3, 10]}': None,
        '{"point_2d": [1.5, 0.2]}': (0.0015, 0.0002),    # a decimal above 1 is on the 0-1000 scale
    }
    for text, want in cases.items():
        for parse in (common.parse_point, adapter.parse_response):
            got = parse(text)
            if want is None:
                assert got is None, (text, got)
            else:
                assert got is not None and all(abs(a - b) < 1e-9 for a, b in zip(got, want)), (text, got, want)


def test_parse_box_center():
    assert common.parse_point_style('{"bbox_2d": [0, 0, 2, 2]}', "bbox") == (0.001, 0.001)
    assert common.parse_point_style('{"bbox_2d": [0.1, 0.1, 0.3, 0.5]}', "bbox") == (0.2, 0.3)
    assert common.parse_point_style('{"bbox_2d": [0, 0, 1200, 10]}', "bbox") is None


if __name__ == "__main__":
    test_prompt_matches()
    test_resize_matches()
    test_parse_matches()
    test_parse_expected_points()
    test_parse_box_center()
    print("adapter matches deskmind_eyes.common")
