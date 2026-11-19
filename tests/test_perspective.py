from dataclasses import replace
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from display_logger import cli
from display_logger.config import Box, Perspective, Profile
from display_logger.diagnostics import save_diagnostics
from display_logger.geometry import perspective_from_points, rectify, roi_polygon, tracking_bounds
from display_logger.synthetic import scene
from display_logger.vision import DisplayReader


def tilted_scene():
    original, profile = scene("24.1")
    height, width = original.shape[:2]
    corners = np.float32([[115, 65], [570, 30], [545, 300], [60, 265]])
    src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    image = cv2.warpPerspective(original, cv2.getPerspectiveTransform(src, corners), (640, 360),
                                borderValue=(160,) * 3)
    spec = Perspective([tuple(p) for p in (corners / [639, 359]).tolist()], width, height)
    return image, replace(profile, perspective=spec)


def test_projective_display_is_corrected_before_reading_digits_and_decimal(tmp_path):
    image, profile = tilted_scene()
    reading = DisplayReader(profile).read(image)
    assert reading.valid, reading
    assert reading.raw == "24.1"
    assert reading.value == 24.1
    profile.save(tmp_path / "perfil.json")
    assert Profile.load(tmp_path / "perfil.json") == profile


def test_roi_is_selected_in_the_rectified_plane_and_projected_back_to_camera():
    image, profile = tilted_scene()
    plane = rectify(image, profile.perspective)
    assert plane.shape[:2] == (profile.perspective.height, profile.perspective.width)
    projected = roi_polygon(image, profile)
    expected = np.array(profile.perspective.corners) * [639, 359]
    assert np.allclose(projected, expected, atol=.01)
    # O rastreador usa o visor na câmera, não o retângulo normalizado da ROI frontal.
    assert tracking_bounds(image, profile) != profile.roi.pixels(640, 360)


@pytest.mark.parametrize("corners", [
    [(0, 0), (1, 1), (1, 0), (0, 1)],
    [(0, 0), (1, 0), (1, 0), (0, 1)],
    [(0, 0), (.5, 0), (1, 0), (0, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 0)],
    [(0, 0), (1.1, 0), (1, 1), (0, 1)],
    [(0, 0), (float("nan"), 0), (1, 1), (0, 1)],
])
def test_invalid_corners_cannot_create_a_perspective(corners):
    with pytest.raises(ValueError):
        Perspective(corners, 400, 150)


def test_calibration_selects_roi_and_cells_on_the_corrected_image(tmp_path, monkeypatch):
    image, expected = tilted_scene()
    monkeypatch.setattr(cli, "require_gui", lambda: None)
    monkeypatch.setattr(cli, "reference_frame", lambda _: image)
    monkeypatch.setattr(cli, "select_perspective", lambda _: expected.perspective)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    boxes = iter([expected.roi, *expected.digits, expected.sign,
                  *(p.box for p in expected.decimal_points)])

    def select(current, *args, **kwargs):
        assert current.shape[:2] == (expected.perspective.height, expected.perspective.width)
        return next(boxes)

    monkeypatch.setattr(cli, "select_box", select)
    target = tmp_path / "frontal.json"
    assert cli.main(["calibrate", "--perspective", "--profile", str(target), "--digits", "4",
                     "--unit", "volts", "--no-tracking"]) == 0
    saved = Profile.load(target)
    assert saved.perspective == expected.perspective
    assert cv2.imread(str(tmp_path / "frontal.referencia.png")).shape == image.shape
    assert cv2.imread(str(tmp_path / "frontal.retificado.png")).shape[:2] == (150, 398)
    assert DisplayReader(saved).read(image).value == 24.1


def test_perspective_diagnostics_keep_raw_frame_and_rectified_coordinates(tmp_path):
    image, profile = tilted_scene()
    save_diagnostics(image, profile, tmp_path)
    report = json.loads((tmp_path / "diagnostico.json").read_text())
    assert report["roi_space"] == "rectified"
    assert cv2.imread(str(tmp_path / "original.png")).shape == image.shape
    assert cv2.imread(str(tmp_path / "retificado.png")).shape[:2] == (150, 398)
    assert report["reading"]["value"] == 24.1


def test_cancelled_perspective_selection_preserves_the_existing_profile(tmp_path, monkeypatch):
    image, profile = tilted_scene()
    target = tmp_path / "perfil.json"
    profile.save(target)
    before = target.read_bytes()
    monkeypatch.setattr(cli, "require_gui", lambda: None)
    monkeypatch.setattr(cli, "reference_frame", lambda _: image)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    def cancel(_):
        raise ValueError("Seleção cancelada")

    monkeypatch.setattr(cli, "select_perspective", cancel)
    assert cli.main(["calibrate", "--perspective", "--profile", str(target), "--unit", "volts"]) == 2
    assert target.read_bytes() == before


def test_automatic_rectification_size_is_based_on_selected_edges():
    spec = perspective_from_points([(10, 10), (210, 10), (210, 90), (10, 90)], 240, 120)
    assert (spec.width, spec.height) == (200, 80)


def test_real_rectified_lcd_with_padded_digit_cells_and_decimal_point():
    directory = Path(__file__).parent / "fixtures"
    profile = Profile.load(directory / "lcd-perspectiva-0071.json")
    frame = cv2.imread(str(directory / "lcd-perspectiva-0071.png"))
    reading = DisplayReader(profile).read(frame)
    assert reading.valid, reading
    assert reading.raw == "0.071"
    assert reading.value == .071
