import json

import cv2
import numpy as np

from display_logger.cli import main
from display_logger.diagnostics import save_diagnostics
from display_logger.synthetic import scene


def test_diagnostics_keep_original_and_report_unknown_digit(tmp_path):
    frame, profile = scene("?123")
    save_diagnostics(frame, profile, tmp_path)
    report = json.loads((tmp_path / "diagnostico.json").read_text())
    assert np.array_equal(cv2.imread(str(tmp_path / "original.png")), frame)
    assert report["reading"]["value"] is None
    assert report["digits"][0]["character"] == "?"
    assert [d["character"] for d in report["digits"][1:]] == ["1", "2", "3"]
    assert set(report["digits"][0]["segment_occupancy"]) == set("abcdefg")
    assert (tmp_path / "enquadramento.png").exists()
    assert (tmp_path / "digito-1-sondas.png").exists()


def test_low_contrast_frame_keeps_diagnostic_without_measurement(tmp_path):
    frame, profile = scene("")
    save_diagnostics(frame, profile, tmp_path)
    report = json.loads((tmp_path / "diagnostico.json").read_text())
    assert report["reading"]["value"] is None
    assert "contraste" in report["preprocessing_error"]
    assert (tmp_path / "original.png").exists()


def test_read_image_saves_segment_diagnostics_on_failure(tmp_path):
    frame, profile = scene("?123")
    profile.save(tmp_path / "perfil.json")
    cv2.imwrite(str(tmp_path / "frame.png"), frame)
    assert main(["read-image", "--profile", str(tmp_path / "perfil.json"),
                 "--image", str(tmp_path / "frame.png"),
                 "--debug-dir", str(tmp_path / "diagnostico")]) == 2
    report = json.loads((tmp_path / "diagnostico" / "diagnostico.json").read_text())
    assert report["digits"][0]["pattern"] == [1, 1, 0, 0, 0, 0, 0]
