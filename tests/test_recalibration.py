from dataclasses import replace

import cv2
import pytest

from display_logger import cli
from display_logger.config import Box, Profile
from display_logger.synthetic import scene


def test_recalibration_reuses_optical_settings_and_updates_selected_regions(tmp_path, monkeypatch):
    frame, original = scene("24.1")
    original = replace(original, digit_slant=.2, otsu_offset=-3,
                       segment_probes=[Box(.3, .02, .4, .05)] * 7)
    source, target = tmp_path / "origem.json", tmp_path / "novo.json"
    original.save(source)
    selected = iter([original.roi, *original.digits, original.sign,
                     *(p.box for p in original.decimal_points)])
    monkeypatch.setattr(cli, "require_gui", lambda: None)
    monkeypatch.setattr(cli, "reference_frame", lambda _: frame)
    monkeypatch.setattr(cli, "select_box", lambda *a, **kw: next(selected))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)
    assert cli.main(["calibrate", "--profile", str(target), "--from-profile", str(source),
                     "--digits", "4", "--unit", "volts"]) == 0
    saved = Profile.load(target)
    assert saved.digit_slant == .2
    assert saved.otsu_offset == -3
    assert saved.segment_probes == original.segment_probes
    assert saved.roi == original.roi
    assert saved.reference_image == "novo.referencia.png"
    assert (tmp_path / saved.reference_image).exists()
    assert Profile.load(source) == original


def test_recalibration_rejects_a_different_recognition_engine_before_opening_gui(tmp_path, monkeypatch):
    _, original = scene("24.1")
    original.save(tmp_path / "origem.json")
    monkeypatch.setattr(cli, "require_gui", lambda: pytest.fail("Não deveria abrir GUI"))
    assert cli.main(["calibrate", "--profile", str(tmp_path / "novo.json"),
                     "--from-profile", str(tmp_path / "origem.json"), "--engine", "tesseract",
                     "--unit", "volts"]) == 2
