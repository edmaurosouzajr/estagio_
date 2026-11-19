from types import SimpleNamespace

import numpy as np
import pytest

from display_logger.yolo_data import (CLASSES, dataset_status, digest, labels_path, prepare_dataset,
                                     read_labels, reviewed_samples, save_review, save_sample)
from display_logger.yolo_training import checked_dataset, evaluate_model, train_model
from display_logger.vision import Reading


def annotation():
    width, height = 310, 60
    labels = []
    for i in range(10):
        x0, x1 = 12 + i * 28, 32 + i * 28
        labels.append((i, (x0 + x1) / (2 * width), .5, (x1 - x0) / width, 40 / height))
    labels.append((10, 35.5 / width, 48.5 / height, 3 / width, 3 / height))
    labels.append((11, 5 / width, 29.5 / height, 8 / width, 3 / height))
    return labels


def sessions(root, all_classes=True):
    images = []
    for index, session in enumerate(("bancada-01", "bancada-02", "bancada-03")):
        frame = np.full((60, 310, 3), 100 + index * 20, dtype=np.uint8)
        image = save_sample(root, session, frame)
        labels = annotation() if all_classes else [(0, .3, .5, .1, .5)]
        save_review(image, labels)
        images.append(image)
    return images


def test_pending_images_do_not_become_negative_training_examples(tmp_path):
    save_sample(tmp_path, "primeira", np.zeros((60, 310, 3), np.uint8))
    result = dataset_status(tmp_path)
    assert result["pending_images"] == 1
    assert result["negative_images"] == 0
    assert result["reviewed_images"] == 0


def test_explicit_negative_image_has_an_empty_label_file(tmp_path):
    image = save_sample(tmp_path, "primeira", np.zeros((60, 310, 3), np.uint8))
    save_review(image, [], negative=True)
    assert labels_path(image).read_text() == ""
    assert dataset_status(tmp_path)["negative_images"] == 1


def test_preparation_keeps_whole_capture_sessions_in_one_split(tmp_path):
    root, target = tmp_path / "source", tmp_path / "prepared"
    sessions(root)
    result = prepare_dataset(root, target, "bancada-02", "bancada-03")
    assert result["sessions"] == {"bancada-01": "train", "bancada-02": "val", "bancada-03": "test"}
    assert result["missing_train_classes"] == []
    _, checked = checked_dataset(target / "data.yaml")
    assert checked["missing_train_classes"] == []
    for image in result["images"]:
        assert image["split"] == result["sessions"][image["session"]]
        assert image["text"] == "-0.123456789"


def test_duplicate_frames_cannot_cross_train_and_test(tmp_path):
    for session in ("a", "b", "c"):
        image = save_sample(tmp_path, session, np.zeros((60, 310, 3), np.uint8))
        save_review(image, annotation())
    with pytest.raises(ValueError, match="idênticas"):
        prepare_dataset(tmp_path, tmp_path / "prepared", "b", "c")


def test_changed_labels_must_be_reviewed_again(tmp_path):
    image = sessions(tmp_path)[0]
    labels_path(image).write_text("0 0.5 0.5 0.1 0.5\n")
    with pytest.raises(ValueError, match="mudou após"):
        reviewed_samples(tmp_path)


def test_prepared_labels_cannot_change_unnoticed(tmp_path):
    root, target = tmp_path / "raw", tmp_path / "prepared"
    sessions(root)
    prepare_dataset(root, target)
    label = next((target / "labels" / "train").glob("*.txt"))
    label.write_text("")
    with pytest.raises(ValueError, match="mudou depois"):
        checked_dataset(target / "data.yaml")


@pytest.mark.parametrize("line", ["12 .5 .5 .1 .1", "0 nan .5 .1 .1", "0 .9 .5 .4 .4", "1 2 3"])
def test_bad_label_coordinates_are_rejected(tmp_path, line):
    path = tmp_path / "bad.txt"
    path.write_text(line)
    with pytest.raises(ValueError):
        read_labels(path)


def test_training_is_blocked_when_a_declared_character_has_no_examples(tmp_path):
    root, target = tmp_path / "raw", tmp_path / "prepared"
    sessions(root, all_classes=False)
    prepare_dataset(root, target)
    with pytest.raises(ValueError, match="Faltam exemplos"):
        train_model(SimpleNamespace(data=target / "data.yaml"))


def test_training_disables_flips_that_would_change_digit_meanings(tmp_path, monkeypatch):
    import display_logger.yolo_training as module

    root, target = tmp_path / "raw", tmp_path / "prepared"
    sessions(root)
    prepare_dataset(root, target)
    calls = []
    best = tmp_path / "trained.pt"

    class Model:
        def __init__(self, *args, **kwargs):
            self.trainer = SimpleNamespace(best=best)

        def train(self, **kwargs):
            calls.append(kwargs)
            best.write_bytes(b"fake checkpoint for isolated test")

    monkeypatch.setattr(module, "ultralytics_yolo", lambda: Model)
    output = train_model(SimpleNamespace(data=target / "data.yaml", output=tmp_path / "best.pt", imgsz=640,
                                        model="fake.pt", epochs=1, batch=2, device="cpu", project=tmp_path / "runs"))
    assert output.is_file()
    assert calls[0]["fliplr"] == calls[0]["flipud"] == calls[0]["mosaic"] == 0
    assert str(target / "data.yaml") == calls[0]["data"]


def test_dataset_session_names_cannot_escape_the_dataset_root(tmp_path):
    with pytest.raises(ValueError):
        save_sample(tmp_path, "../outside", np.zeros((60, 310, 3), np.uint8))


def test_evaluation_counts_complete_readings_and_negative_false_accepts_separately(tmp_path, monkeypatch):
    import display_logger.yolo_training as module

    root, target = tmp_path / "raw", tmp_path / "prepared"
    sessions(root)
    negative = save_sample(root, "bancada-03", np.zeros((60, 310, 3), np.uint8))
    save_review(negative, [], negative=True)
    prepare_dataset(root, target, "bancada-02", "bancada-03")

    class Reader:
        def __init__(self, profile):
            pass

        def read(self, image):
            return Reading("0.000", 0., .9) if image.mean() == 0 else Reading("-0.123456789", -.123456789, .9)

    monkeypatch.setattr(module, "DisplayReader", Reader)
    result = evaluate_model(SimpleNamespace(data=target / "data.yaml", weights="fake.pt", imgsz=640,
                                            device="cpu", expected_digits=10, allow_integer=False,
                                            output=tmp_path / "evaluation.json"))
    assert result["positive_images"] == result["negative_images"] == 1
    assert result["complete_reading_accuracy"] == 1
    assert result["negative_false_accepts"] == 1
