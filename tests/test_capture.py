from argparse import Namespace

import cv2
import pytest

from display_logger import cli


class FakeCapture:
    def __init__(self, opened=True, backend="V4L2"):
        self.opened = opened
        self.backend = backend
        self.released = False
        self.properties = []

    def isOpened(self):
        return self.opened

    def release(self):
        self.released = True

    def getBackendName(self):
        return self.backend

    def set(self, prop, value):
        self.properties.append((prop, value))
        return True


def args(**changes):
    return Namespace(**{"camera": 0, "video": None, "camera_backend": "auto",
                        "width": 1280, "height": 720, **changes})


def captures(monkeypatch, objects, platform="linux"):
    calls = []
    iterator = iter(objects)

    def create(source, backend):
        calls.append((source, backend))
        return next(iterator)

    monkeypatch.setattr(cli.sys, "platform", platform)
    monkeypatch.setattr(cv2, "VideoCapture", create)
    return calls


def test_linux_prefers_v4l2_for_local_webcam(monkeypatch):
    camera = FakeCapture()
    calls = captures(monkeypatch, [camera])
    assert cli.open_capture(args()) is camera
    assert calls == [(0, cv2.CAP_V4L2)]
    assert (cv2.CAP_PROP_BUFFERSIZE, 1) in camera.properties


def test_fallback_releases_v4l2_and_avoids_unsupported_gstreamer_buffer(monkeypatch):
    unavailable = FakeCapture(opened=False)
    camera = FakeCapture(backend="GSTREAMER")
    calls = captures(monkeypatch, [unavailable, camera])
    assert cli.open_capture(args()) is camera
    assert calls == [(0, cv2.CAP_V4L2), (0, cv2.CAP_ANY)]
    assert unavailable.released
    assert all(prop != cv2.CAP_PROP_BUFFERSIZE for prop, _ in camera.properties)


def test_explicit_gstreamer_does_not_try_v4l2(monkeypatch):
    camera = FakeCapture(backend="GSTREAMER")
    calls = captures(monkeypatch, [camera])
    assert cli.open_capture(args(camera_backend="gstreamer")) is camera
    assert calls == [(0, cv2.CAP_GSTREAMER)]


def test_explicit_v4l2_failure_does_not_silently_change_backend(monkeypatch):
    unavailable = FakeCapture(opened=False)
    calls = captures(monkeypatch, [unavailable])
    with pytest.raises(ValueError, match="Não foi possível abrir"):
        cli.open_capture(args(camera_backend="v4l2"))
    assert calls == [(0, cv2.CAP_V4L2)]
    assert unavailable.released


def test_video_keeps_automatic_backend_and_does_not_set_camera_properties(monkeypatch):
    video = FakeCapture(backend="FFMPEG")
    calls = captures(monkeypatch, [video])
    assert cli.open_capture(args(video="ensaio.avi")) is video
    assert calls == [("ensaio.avi", cv2.CAP_ANY)]
    assert video.properties == []


def test_other_platforms_keep_opencv_automatic_selection(monkeypatch):
    camera = FakeCapture(backend="MSMF")
    calls = captures(monkeypatch, [camera], platform="win32")
    assert cli.open_capture(args()) is camera
    assert calls == [(0, cv2.CAP_ANY)]


def test_exhausted_backends_release_all_failed_captures(monkeypatch):
    cameras = [FakeCapture(opened=False), FakeCapture(opened=False)]
    captures(monkeypatch, cameras)
    with pytest.raises(ValueError, match="Não foi possível abrir"):
        cli.open_capture(args())
    assert all(camera.released for camera in cameras)
