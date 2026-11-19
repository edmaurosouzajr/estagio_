"""Alinhamento com a moldura, excluindo os dígitos que mudam durante o teste."""

import cv2
import numpy as np
from .geometry import tracking_bounds


def edges(gray):
    return cv2.GaussianBlur(cv2.Canny(gray, 40, 100), (5, 5), 1.2)


class DisplayTracker:
    def __init__(self, profile, reference):
        self.profile = profile
        self.reference = reference
        self.shape = None
        self.last_match = None

    def _prepare(self, frame):
        height, width = frame.shape[:2]
        reference = cv2.resize(self.reference, (width, height))
        x0, y0, x1, y1 = tracking_bounds(frame, self.profile)
        pad_x, pad_y = max(12, round((x1 - x0) * .25)), max(12, round((y1 - y0) * .5))
        self.rect = (max(0, x0 - pad_x), max(0, y0 - pad_y),
                     min(width, x1 + pad_x), min(height, y1 + pad_y))
        left, top, right, bottom = self.rect
        self.gray = cv2.cvtColor(reference[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        self.mask = np.full_like(self.gray, 255)
        # A moldura e a etiqueta fixas orientam a busca, não o valor mostrado.
        self.mask[max(0, y0 - top - 1):y1 - top + 1,
                  max(0, x0 - left - 1):x1 - left + 1] = 0
        self.template = edges(self.gray)
        if np.count_nonzero(self.template[self.mask != 0]) < 30:
            raise ValueError("Referência sem contorno suficiente ao redor do display; refaça a calibração.")
        self.shape = frame.shape[:2]

    def align(self, frame):
        if self.shape != frame.shape[:2]:
            self._prepare(frame)
        height, width = frame.shape[:2]
        left, top, right, bottom = self.rect
        radius = max(24, round(max(width, height) * .06))
        sx0, sy0 = max(0, left - radius), max(0, top - radius)
        sx1, sy1 = min(width, right + radius), min(height, bottom + radius)
        search = cv2.cvtColor(frame[sy0:sy1, sx0:sx1], cv2.COLOR_BGR2GRAY)
        scores = cv2.matchTemplate(edges(search), self.template, cv2.TM_CCORR_NORMED, mask=self.mask)
        scores = np.nan_to_num(scores, nan=-1., posinf=-1., neginf=-1.)
        _, best, _, where = cv2.minMaxLoc(scores)
        other = scores.copy()
        dx, dy = max(10, self.gray.shape[1] // 9), max(10, self.gray.shape[0] // 7)
        x, y = where
        other[max(0, y - dy):y + dy + 1, max(0, x - dx):x + dx + 1] = -1
        if best < .65 or best - float(other.max()) < .05:
            self.last_match = None
            raise ValueError("Display não localizado com confiança; mantenha a moldura visível ou recalibre.")
        cx, cy = sx0 + x, sy0 + y
        current = cv2.cvtColor(frame[cy:cy + bottom - top, cx:cx + right - left], cv2.COLOR_BGR2GRAY)
        try:
            correlation, local = cv2.findTransformECC(
                self.gray, current, np.eye(2, 3, dtype=np.float32), cv2.MOTION_AFFINE,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5), self.mask, 5,
            )
        except cv2.error as exc:
            self.last_match = None
            raise ValueError("Não foi possível alinhar a moldura do display; confira o enquadramento.") from exc
        if not np.isfinite(local).all() or not np.isfinite(correlation):
            self.last_match = None
            raise ValueError("Alinhamento inválido; confira o enquadramento do display.")
        scales = np.linalg.svd(local[:, :2], compute_uv=False)
        if (correlation < .90 or np.linalg.det(local[:, :2]) <= 0 or min(scales) < .85 or max(scales) > 1.15
                or max(abs(local[0, 1]), abs(local[1, 0])) > .15
                or abs(local[0, 2]) > self.gray.shape[1] * .2
                or abs(local[1, 2]) > self.gray.shape[0] * .2):
            self.last_match = None
            raise ValueError("Mudança de posição ou perspectiva excede o rastreamento; recalibre o display.")
        matrix = local.copy()
        matrix[:, 2] += np.array([cx, cy]) - local[:, :2] @ np.array([left, top])
        x0, y0, x1, y1 = tracking_bounds(frame, self.profile)
        corners = np.array([[x0, y0, 1], [x1, y0, 1], [x1, y1, 1], [x0, y1, 1]]) @ matrix.T
        if (corners[:, 0].min() < 0 or corners[:, 1].min() < 0
                or corners[:, 0].max() > width or corners[:, 1].max() > height):
            raise ValueError("O display saiu do quadro da webcam.")
        self.last_match = {"template_score": float(best), "alignment_score": float(correlation),
                           "transform": matrix.tolist()}
        return cv2.warpAffine(frame, matrix, (width, height),
                              flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
