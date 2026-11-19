import json

import cv2
import numpy as np

from display_logger.cli import main
from display_logger.synthetic import scene


def test_demo_and_single_frame_commands(tmp_path, capsys):
    assert main(["demo", "--out-dir", str(tmp_path)]) == 0
    records = [json.loads(line) for line in (tmp_path / "leituras.txt").read_text().splitlines()]
    assert [record["value"] for record in records] == [24.1, -.08, 120.0, 12.34, None, None]
    assert [record["status"] for record in records] == ["ok"] * 4 + ["invalid"] * 2
    assert main(["read-image", "--profile", str(tmp_path / "perfil.json"),
                 "--image", str(tmp_path / "display-1.png")]) == 0
    assert main(["read-image", "--profile", str(tmp_path / "perfil.json"),
                 "--image", str(tmp_path / "display-5.png")]) == 2


def test_camera_loop_logs_fresh_values_and_closes_capture(tmp_path, monkeypatch):
    import display_logger.cli as cli

    image, profile = scene("24.1")
    profile.save(tmp_path / "perfil.json")
    # Quadro inválido depois da estabilização deve limpar o valor imediatamente.
    frames = [image] * 4 + [np.full_like(image, 220)]

    class Camera:
        released = False

        def read(self):
            return True, frames.pop(0)

        def release(self):
            self.released = True

    camera = Camera()
    monkeypatch.setattr(cli, "open_capture", lambda _: camera)
    ticks = iter(i * .1 for i in range(100))
    monkeypatch.setattr(cli.time, "monotonic", lambda: next(ticks))
    assert main(["run", "--profile", str(tmp_path / "perfil.json"), "--test-id", "T-001",
                 "--out", str(tmp_path / "log.txt"), "--no-preview", "--samples", "5",
                 "--interval", "0.01"]) == 0
    records = [json.loads(line) for line in (tmp_path / "log.txt").read_text().splitlines()]
    assert [r["status"] for r in records] == ["unstable", "unstable", "ok", "ok", "invalid"]
    assert [r["value"] for r in records] == [None, None, 24.1, 24.1, None]
    assert camera.released


def test_real_ocr_pipeline_when_tesseract_is_available():
    import shutil
    import pytest
    from display_logger.config import Box, Profile
    from display_logger.vision import DisplayReader

    if not shutil.which("tesseract"):
        pytest.skip("Tesseract não instalado; integração opcional")
    image = np.full((130, 450, 3), 255, dtype=np.uint8)
    cv2.putText(image, "24.1", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 5)
    profile = Profile(name="ocr-teste", unit="volts", roi=Box(0, 0, 1, 1), engine="tesseract")
    reading = DisplayReader(profile).read(image)
    assert reading.valid, reading
    assert reading.value == 24.1
