from pathlib import Path
import json
import numpy as np

import cv2

from display_logger.cli import main
from display_logger.config import Profile
from display_logger.synthetic import scene
from display_logger.yolo_data import dataset_status


def test_existing_calibration_can_create_a_separate_yolo_profile_before_training(tmp_path):
    frame, source = scene("24.1")
    source.save(tmp_path / "source.json")
    cv2.imwrite(str(tmp_path / "reference.png"), frame)
    assert main(["yolo-profile", "--from-profile", str(tmp_path / "source.json"),
                 "--profile", str(tmp_path / "yolo.json"), "--weights", str(tmp_path / "future.pt")]) == 0
    target = Profile.load(tmp_path / "yolo.json")
    assert target.engine == "yolo"
    assert target.roi == source.roi
    assert target.require_decimal
    assert target.yolo_expected_digits == 4
    assert target.digits == []
    assert Profile.load(tmp_path / "source.json").engine == "seven_segment"
    assert main(["read-image", "--profile", str(tmp_path / "yolo.json"),
                 "--image", str(tmp_path / "reference.png")]) == 2


def test_imported_images_remain_unreviewed_until_human_annotation(tmp_path):
    frame, _ = scene("24.1")
    image = tmp_path / "source.png"
    cv2.imwrite(str(image), frame)
    root = tmp_path / "dataset"
    assert main(["yolo-import", "--images", str(image), "--dataset", str(root), "--session", "bancada-01"]) == 0
    report = dataset_status(root)
    assert report["pending_images"] == 1
    assert report["reviewed_images"] == 0
    assert main(["yolo-prepare", "--dataset", str(root), "--output", str(tmp_path / "prepared")]) == 2


def test_yolo_geometric_calibration_does_not_ask_for_digit_cells_or_require_weights(tmp_path, monkeypatch):
    import display_logger.cli as cli

    frame, profile = scene("24.1")
    profile.save(tmp_path / "source.json")
    selections = []
    monkeypatch.setattr(cli, "require_gui", lambda: None)
    monkeypatch.setattr(cli, "reference_frame", lambda _: frame)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    def choose(*args, **kwargs):
        selections.append(True)
        return profile.roi

    monkeypatch.setattr(cli, "select_box", choose)
    assert main(["calibrate", "--engine", "yolo", "--from-profile", str(tmp_path / "source.json"),
                 "--profile", str(tmp_path / "yolo.json"), "--unit", "volts",
                 "--weights", str(tmp_path / "future.pt")]) == 0
    saved = Profile.load(tmp_path / "yolo.json")
    assert len(selections) == 1
    assert saved.digits == []
    assert saved.yolo_expected_digits == 4
    assert saved.require_decimal


def test_video_collection_samples_source_time_and_releases_the_capture(tmp_path, monkeypatch):
    import display_logger.cli as cli

    frame, profile = scene("24.1")
    profile.save(tmp_path / "profile.json")

    class Capture:
        index = 0
        released = False

        def get(self, prop):
            return 10.

        def read(self):
            image = np.clip(frame.astype(int) + self.index, 0, 255).astype("uint8")
            self.index += 1
            return self.index <= 31, image

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(cli, "open_capture", lambda _: capture)
    dataset = tmp_path / "dataset"
    assert main(["yolo-collect", "--video", "fake.avi", "--profile", str(tmp_path / "profile.json"),
                 "--dataset", str(dataset), "--session", "video-01", "--samples", "3", "--interval", "1",
                 "--no-preview"]) == 0
    metadata = [json.loads(p.read_text()) for p in (dataset / "raw" / "video-01" / "images").glob("*.source.json")]
    assert sorted(m["source_seconds"] for m in metadata) == [0., 1., 2.]
    assert capture.released
