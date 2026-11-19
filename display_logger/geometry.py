"""Perspectiva do visor; ROI e dígitos pertencem ao plano retificado."""

import cv2
import numpy as np

from .config import Perspective


def perspective_from_points(points, width, height):
    points = np.asarray(points, dtype=np.float32)
    if points.shape != (4, 2) or width < 2 or height < 2:
        raise ValueError("Informe quatro cantos em uma imagem válida.")
    normalized = points / np.array([width - 1, height - 1])
    out_width = round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))
    out_height = round(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))
    return Perspective([tuple(p) for p in normalized.tolist()], out_width, out_height)


def source_corners(frame, spec):
    return np.asarray(spec.corners, dtype=np.float32) * np.array(
        [frame.shape[1] - 1, frame.shape[0] - 1], dtype=np.float32)


def homography(frame, spec):
    destination = np.float32([[0, 0], [spec.width - 1, 0],
                             [spec.width - 1, spec.height - 1], [0, spec.height - 1]])
    matrix = cv2.getPerspectiveTransform(source_corners(frame, spec), destination)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Não foi possível corrigir a perspectiva; selecione novamente os cantos.")
    return matrix


def rectify(frame, spec):
    if spec is None:
        return frame
    return cv2.warpPerspective(frame, homography(frame, spec), (spec.width, spec.height),
                               flags=cv2.INTER_LINEAR)


def display_image(frame, profile):
    return profile.roi.crop(rectify(frame, profile.perspective))


def roi_polygon(frame, profile):
    spec = profile.perspective
    width, height = (spec.width, spec.height) if spec else (frame.shape[1], frame.shape[0])
    x0, y0, x1, y1 = profile.roi.pixels(width, height)
    corners = np.float32([[x0, y0], [x1 - 1, y0], [x1 - 1, y1 - 1], [x0, y1 - 1]])
    if spec:
        corners = cv2.perspectiveTransform(corners[None], np.linalg.inv(homography(frame, spec)))[0]
    return corners


def tracking_bounds(frame, profile):
    if profile.perspective is None:
        return profile.roi.pixels(frame.shape[1], frame.shape[0])
    points = source_corners(frame, profile.perspective)
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int) + 1
    return max(0, x0), max(0, y0), min(frame.shape[1], x1), min(frame.shape[0], y1)
