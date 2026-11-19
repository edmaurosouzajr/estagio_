from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from display_logger.config import Profile
from display_logger.synthetic import scene
from display_logger.vision import DisplayReader


@pytest.mark.parametrize("name,expected", [
    ("lcd-0009", "0.009"),
    ("lcd-perspectiva-0071", "0.071"),
    ("lcd-perspectiva-0061", "0.061"),
])
def test_search_reads_different_real_lcd_values_without_changing_their_digits(name, expected):
    folder = Path(__file__).parent / "fixtures"
    profile = replace(Profile.load(folder / f"{name}.json"), segment_search=.06)
    frame = cv2.imread(str(folder / f"{name}.png"))
    reading = DisplayReader(profile).read(frame)
    assert reading.valid, reading
    assert reading.raw == expected


def test_real_six_with_center_stroke_between_the_old_probe_rows():
    folder = Path(__file__).parent / "fixtures"
    profile = Profile.load(folder / "lcd-perspectiva-0061.json")
    frame = cv2.imread(str(folder / "lcd-perspectiva-0061.png"))
    assert not DisplayReader(replace(profile, segment_search=0)).read(frame).valid
    assert DisplayReader(replace(profile, segment_search=.06)).read(frame).raw == "0.061"


@pytest.mark.parametrize("polarity", ["dark", "light"])
def test_search_does_not_change_the_ten_valid_digit_patterns(polarity):
    frame, profile = scene("0123456789", count=10, decimal_mode="none", polarity=polarity)
    reading = DisplayReader(replace(profile, segment_search=.06)).read(frame)
    assert reading.valid, reading
    assert reading.raw == "0123456789"


@pytest.mark.parametrize("text", ["?123", "OL", ""])
def test_search_still_rejects_unknown_patterns_and_empty_displays(text):
    frame, profile = scene(text)
    reading = DisplayReader(replace(profile, segment_search=.06)).read(frame)
    assert not reading.valid
    assert reading.value is None
