from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from display_logger.cli import main, read_frame
from display_logger.config import Box, Perspective, Profile
from display_logger.tracking import DisplayTracker
from display_logger.vision import DisplayReader, StabilityGate


FIXTURES = Path(__file__).parent / "fixtures"


def captures():
    profile = Profile.load(FIXTURES / "lcd-tracking.json")
    reference = cv2.imread(str(FIXTURES / "lcd-reference-context.png"))
    moved = cv2.imread(str(FIXTURES / "lcd-moved-context.png"))
    return profile, reference, moved


def test_real_lcd_motion_is_aligned_independently_of_the_displayed_value():
    profile, reference, moved = captures()
    tracker = DisplayTracker(profile, reference)
    reader = DisplayReader(profile)
    original = reader.read(tracker.align(reference))
    changed = reader.read(tracker.align(moved))
    assert original.valid, original
    assert changed.valid, changed
    assert original.raw == "0.009"
    assert changed.raw == "0.000"
    assert tracker.last_match["alignment_score"] >= .90


def test_lost_display_invalidates_the_current_measurement():
    profile, reference, moved = captures()
    reader, tracker, gate = DisplayReader(profile), DisplayTracker(profile, reference), StabilityGate(1)
    _, good = read_frame(reader, tracker, moved)
    assert gate.update(good) == "ok"
    _, lost = read_frame(reader, tracker, np.zeros_like(moved))
    assert lost.value is None
    assert not lost.valid
    assert gate.update(lost) == "invalid"
    assert tracker.last_match is None


def test_tracking_requires_visible_context():
    profile, reference, moved = captures()
    tracker = DisplayTracker(profile, np.zeros_like(reference))
    with pytest.raises(ValueError, match="contorno"):
        tracker.align(moved)


def test_image_command_loads_a_relative_tracking_reference(tmp_path):
    profile, reference, moved = captures()
    cv2.imwrite(str(tmp_path / "referencia.png"), reference)
    cv2.imwrite(str(tmp_path / "atual.png"), moved)
    replace(profile, track_display=True, reference_image="referencia.png").save(tmp_path / "perfil.json")
    assert main(["read-image", "--profile", str(tmp_path / "perfil.json"),
                 "--image", str(tmp_path / "atual.png")]) == 0


def test_tracking_stays_in_camera_coordinates_before_perspective_correction():
    profile, reference, moved = captures()
    height, width = reference.shape[:2]
    x0, y0, x1, y1 = profile.roi.pixels(width, height)
    corners = [(x0 / (width - 1), y0 / (height - 1)),
               ((x1 - 1) / (width - 1), y0 / (height - 1)),
               ((x1 - 1) / (width - 1), (y1 - 1) / (height - 1)),
               (x0 / (width - 1), (y1 - 1) / (height - 1))]
    profile = replace(profile, roi=Box(0, 0, 1, 1),
                      perspective=Perspective(corners, x1 - x0, y1 - y0))
    tracker = DisplayTracker(profile, reference)
    reading = DisplayReader(profile).read(tracker.align(moved))
    assert reading.valid, reading
    assert reading.raw == "0.000"
