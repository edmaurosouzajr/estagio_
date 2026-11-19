from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from display_logger.config import Box, Profile
from display_logger.synthetic import scene
from display_logger.vision import DisplayReader, Reading, StabilityGate, parse_number


@pytest.mark.parametrize("polarity", ["dark", "light"])
@pytest.mark.parametrize("text,expected", [
    ("24.1", 24.1), ("-0.08", -0.08), ("120.0", 120.0),
    ("12.34", 12.34), ("0000", 0), ("1", 1), ("8", 8),
])
def test_rendered_digits_sign_decimal_and_leading_blanks(polarity, text, expected):
    image, profile = scene(text, polarity=polarity)
    reading = DisplayReader(profile).read(image)
    assert reading.valid, reading.reason
    assert reading.value == expected


def test_every_digit_and_normalized_coordinates_at_other_resolution():
    image, profile = scene("0123456789", count=10)
    image = cv2.resize(image, None, fx=1.5, fy=1.5)
    reading = DisplayReader(profile).read(image)
    assert reading.valid, reading.reason
    assert reading.raw == "0123456789"


def test_noisy_frame_with_moderate_illumination_gradient():
    image, profile = scene("24.1")
    rng = np.random.default_rng(12)
    gradient = np.linspace(-20, 10, image.shape[1])[None, :, None]
    image = np.clip(image.astype(float) + gradient + rng.normal(0, 2, image.shape), 0, 255).astype("uint8")
    reading = DisplayReader(profile).read(image)
    assert reading.valid, reading.reason
    assert reading.value == 24.1


@pytest.mark.parametrize("text", ["OL", "", "?123", "1 23"])
def test_unknown_patterns_blank_frames_and_gaps_are_rejected(text):
    image, profile = scene(text)
    reading = DisplayReader(profile).read(image)
    assert not reading.valid
    assert reading.value is None


def test_multiple_points_are_rejected():
    image, profile = scene("12.34")
    x0, y0, x1, y1 = profile.decimal_points[0].box.pixels(image.shape[1], image.shape[0])
    image[y0:y1, x0:x1] = 25
    reading = DisplayReader(profile).read(image)
    assert not reading.valid
    assert "Vários pontos" in reading.reason


def test_required_decimal_rejects_a_missing_point_instead_of_inventing_scale():
    image, profile = scene("241")
    reading = DisplayReader(replace(profile, require_decimal=True)).read(image)
    assert not reading.valid
    assert "ponto decimal" in reading.reason


def test_fixed_scale_is_explicit_and_allows_leading_zeroes():
    image, profile = scene("0008", decimal_mode="fixed", decimal_places=2)
    reading = DisplayReader(profile).read(image)
    assert reading.valid, reading.reason
    assert reading.value == .08


def test_profile_limits_and_low_quality_are_rejected():
    image, profile = scene("24.1")
    assert not DisplayReader(replace(profile, value_max=20)).read(image).valid
    assert not DisplayReader(replace(profile, min_confidence=.95)).read(image).valid


@pytest.mark.parametrize("text", ["OL", "NaN", "inf", "1.2.3", "24 1", "24V", "", "--2", "1e4"])
def test_numeric_parser_does_not_guess(text):
    with pytest.raises(ValueError):
        parse_number(text)


def test_numeric_parser_handles_decimal_comma_and_unicode_minus():
    assert parse_number("−24,1") == -24.1


def test_stability_never_reuses_a_stale_value():
    gate = StabilityGate(3)
    good = Reading("24.1", 24.1, .9)
    bad = Reading("?", None, .1, "Imagem inválida")
    assert [gate.update(good) for _ in range(3)] == ["unstable", "unstable", "ok"]
    assert gate.update(bad) == "invalid"
    assert gate.update(good) == "unstable"
    assert gate.update(Reading("24.2", 24.2, .9)) == "unstable"


def test_profile_roundtrip_and_invalid_coordinates(tmp_path):
    _, profile = scene("24.1")
    target = tmp_path / "perfil.json"
    profile.save(target)
    assert Profile.load(target) == profile
    with pytest.raises(ValueError):
        Box(.9, 0, .2, .5)
    with pytest.raises(ValueError):
        Box(float("nan"), 0, .2, .5)
    with pytest.raises(ValueError):
        replace(profile, decimal_points=[])


def test_ambiguous_segment_is_not_rounded_to_a_digit():
    image, profile = scene("8888")
    digit = profile.digits[0]
    x0, y0, x1, y1 = digit.pixels(image.shape[1], image.shape[0])
    # Apaga aproximadamente metade da sonda do segmento superior.
    image[y0:y0 + 15, x0 + 25:x1] = 220
    reading = DisplayReader(profile).read(image)
    assert not reading.valid


def test_ocr_preserves_spaces_and_rejects_status_text(monkeypatch):
    import subprocess
    import display_logger.vision as vision

    image, profile = scene("24.1")
    profile = replace(profile, engine="tesseract")
    monkeypatch.setattr(vision.shutil, "which", lambda _: "/fake/tesseract")
    tsv = b"level\tconf\ttext\n5\t98\t24\n5\t97\t1\n"
    monkeypatch.setattr(vision.subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, tsv, b""))
    reading = DisplayReader(profile).read(image)
    assert reading.raw == "24 1"
    assert not reading.valid


@pytest.mark.parametrize("direction", [1, -1])
def test_slanted_cells_are_rectified_before_decoding_all_digits(direction):
    source, profile = scene("0123456789", count=10, decimal_mode="none")
    frame = np.full_like(source, 220)
    cells = []
    shift = 10
    for box in profile.digits:
        x0, y0, x1, y1 = box.pixels(source.shape[1], source.shape[0])
        patch = source[y0:y1, x0:x1]
        height, width = patch.shape[:2]
        matrix = np.array([[1, -direction * shift / (height - 1), max(direction * shift, 0)],
                           [0, 1, 0]], dtype=np.float32)
        slanted = cv2.warpAffine(patch, matrix, (width + shift, height),
                                  flags=cv2.INTER_NEAREST, borderValue=(220,) * 3)
        frame[y0:y1, x0:x0 + width + shift] = slanted
        cells.append(Box.from_pixels((x0, y0, width + shift, height), source.shape[1], source.shape[0]))
    adjusted = replace(profile, digits=cells, digit_slant=direction * shift / 60)
    reading = DisplayReader(adjusted).read(frame)
    assert reading.valid, reading
    assert reading.raw == "0123456789"


def test_captured_lcd_with_italic_segments_and_decimal_point():
    folder = Path(__file__).parent / "fixtures"
    profile = Profile.load(folder / "lcd-0009.json")
    frame = cv2.imread(str(folder / "lcd-0009.png"))
    reading = DisplayReader(profile).read(frame)
    assert reading.valid, reading
    assert reading.raw == "0.009"
    assert reading.value == .009
    assert reading.confidence >= profile.min_confidence


def test_geometry_configuration_rejects_invalid_settings():
    _, profile = scene("24.1")
    with pytest.raises(ValueError):
        replace(profile, digit_slant=.8)
    with pytest.raises(ValueError):
        replace(profile, threshold="adaptive", otsu_offset=-3)
    with pytest.raises(ValueError):
        replace(profile, segment_probes=[Box(0, 0, 1, 1)])
