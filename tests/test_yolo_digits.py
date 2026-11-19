from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from display_logger.config import Box, Profile
from display_logger.vision import DisplayReader
from display_logger.yolo_digits import CLASSES, Detection, assemble


def characters(text="0071", decimal=True, minus=False):
    detections = [Detection(c, .9, (12 + i * 28, 10, 32 + i * 28, 50)) for i, c in enumerate(text)]
    if decimal:
        detections.append(Detection("dot", .85, (34, 47, 37, 50)))
    if minus:
        detections.append(Detection("minus", .9, (1, 28, 9, 31)))
    return detections


def fake_factory(monkeypatch, detections, names=None):
    import display_logger.yolo_digits as module

    array = np.array([[*d.xyxy, d.confidence, CLASSES.index(d.label)] for d in detections], dtype=float)
    tensor = SimpleNamespace(numpy=lambda: array)
    tensor.cpu = lambda: tensor
    calls = []

    class Model:
        def __init__(self, weights, task):
            calls.append((weights, task))
            self.names = names or dict(enumerate(CLASSES))

        def predict(self, **kwargs):
            return [SimpleNamespace(boxes=SimpleNamespace(data=tensor))]

    monkeypatch.setattr(module, "ultralytics_yolo", lambda: Model)
    return calls


def test_characters_are_sorted_with_decimal_and_negative_sign():
    detections = characters(minus=True)
    raw, score = assemble(list(reversed(detections)), expected_digits=4)
    assert raw == "-0.071"
    assert score == .85


def test_missing_digit_is_rejected_instead_of_logging_an_incomplete_number():
    with pytest.raises(ValueError, match="incompleta"):
        assemble(characters()[:-2], expected_digits=4)


def test_conflicting_digit_predictions_are_rejected():
    detections = characters()
    detections.append(replace(detections[0], label="8"))
    with pytest.raises(ValueError, match="sobrepostas"):
        assemble(detections)


def test_decimal_and_sign_must_have_valid_positions():
    detections = characters()
    detections[-1] = replace(detections[-1], xyxy=(34, 15, 37, 18))
    with pytest.raises(ValueError, match="Ponto decimal"):
        assemble(detections)
    with pytest.raises(ValueError, match="Sinal negativo"):
        assemble(characters() + [Detection("minus", .9, (120, 28, 125, 31))])


def test_two_decimal_points_are_rejected():
    with pytest.raises(ValueError, match="Mais de um"):
        assemble(characters() + [Detection("dot", .9, (62, 47, 65, 50))])


def test_yolo_reader_uses_color_roi_without_segment_thresholding(tmp_path, monkeypatch):
    import display_logger.vision as module

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake only for isolated tests")
    fake_factory(monkeypatch, characters())
    monkeypatch.setattr(module, "preprocess", lambda *a: pytest.fail("YOLO deve receber o recorte colorido"))
    profile = Profile(name="yolo", unit="volts", roi=Box(0, 0, 1, 1), engine="yolo",
                      yolo_weights=str(weights), yolo_expected_digits=4, require_decimal=True)
    reading = DisplayReader(profile).read(np.full((60, 135, 3), 110, np.uint8))
    assert reading.valid, reading
    assert reading.raw == "0.071"
    assert reading.value == .071
    assert len(reading.detections) == 5


def test_low_confidence_decimal_is_not_silently_discarded(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    detections = characters()
    detections[-1] = replace(detections[-1], confidence=.2)
    fake_factory(monkeypatch, detections)
    profile = Profile(name="yolo", unit="volts", roi=Box(0, 0, 1, 1), engine="yolo", yolo_weights=str(weights))
    reading = DisplayReader(profile).read(np.zeros((60, 135, 3), np.uint8))
    assert reading.raw == "0.071"
    assert reading.value is None
    assert "Qualidade" in reading.reason


def test_missing_decimal_is_rejected_when_required(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"fake")
    fake_factory(monkeypatch, characters(decimal=False))
    profile = Profile(name="yolo", unit="volts", roi=Box(0, 0, 1, 1), engine="yolo",
                      yolo_weights=str(weights), require_decimal=True)
    reading = DisplayReader(profile).read(np.zeros((60, 135, 3), np.uint8))
    assert reading.value is None
    assert "ponto decimal" in reading.reason


def test_coco_weights_cannot_masquerade_as_a_digit_model(tmp_path, monkeypatch):
    weights = tmp_path / "coco.pt"
    weights.write_bytes(b"fake")
    fake_factory(monkeypatch, [], names={0: "person"})
    profile = Profile(name="yolo", unit="volts", roi=Box(0, 0, 1, 1), engine="yolo", yolo_weights=str(weights))
    with pytest.raises(ValueError, match="incompatíveis"):
        DisplayReader(profile)


def test_missing_weights_do_not_trigger_automatic_download(tmp_path, monkeypatch):
    import display_logger.yolo_digits as module

    monkeypatch.setattr(module, "ultralytics_yolo", lambda: pytest.fail("Não deve baixar modelo na inferência"))
    profile = Profile(name="yolo", unit="volts", roi=Box(0, 0, 1, 1), engine="yolo",
                      yolo_weights=str(tmp_path / "missing.pt"))
    with pytest.raises(ValueError, match="ainda não encontrados"):
        DisplayReader(profile)
