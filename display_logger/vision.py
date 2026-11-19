"""Decodificador determinístico de segmentos e OCR numérico opcional."""

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import csv
import io
import math
import re
import shutil
import subprocess

import cv2
import numpy as np

from .config import Box, Profile
from .geometry import display_image

# Ordem a, b, c, d, e, f, g: topo, dir. superior, dir. inferior,
# base, esq. inferior, esq. superior, meio.
DIGIT_SEGMENTS = {
    "0": "1111110", "1": "0110000", "2": "1101101", "3": "1111001",
    "4": "0110011", "5": "1011011", "6": "1011111", "7": "1110000",
    "8": "1111111", "9": "1111011", "-": "0000001", " ": "0000000",
}
PATTERNS = {tuple(map(int, bits)): char for char, bits in DIGIT_SEGMENTS.items()}
PROBES = (
    Box(.25, .025, .50, .10), Box(.80, .18, .18, .24),
    Box(.80, .58, .18, .24), Box(.25, .875, .50, .10),
    Box(.02, .58, .18, .24), Box(.02, .18, .18, .24),
    Box(.25, .45, .50, .10),
)
NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:[.,][0-9]*)?|[.,][0-9]+)")


@dataclass(frozen=True)
class Reading:
    raw: str
    value: float | None
    confidence: float
    reason: str = ""
    detections: tuple = ()

    @property
    def valid(self):
        return self.value is not None and not self.reason


def parse_number(text):
    """Não adivinha letras, não remove espaços internos e não aceita vários pontos."""
    text = text.strip().replace("−", "-")
    if not NUMBER.fullmatch(text):
        raise ValueError(f"Leitura não numérica ou ambígua: {text!r}")
    try:
        value = float(Decimal(text.replace(",", ".")))
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError("Número inválido.") from exc
    if not math.isfinite(value):
        raise ValueError("Número fora da faixa representável.")
    return value


def preprocess(frame, profile):
    roi = display_image(frame, profile)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
    if min(gray.shape) < 8:
        raise ValueError("Display pequeno demais; aproxime a câmera ou ajuste a região.")
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    if int(gray.max()) - int(gray.min()) < 15:
        raise ValueError("Imagem sem contraste suficiente.")
    kind = cv2.THRESH_BINARY_INV if profile.polarity == "dark" else cv2.THRESH_BINARY
    if profile.threshold == "adaptive":
        size = min(31, min(gray.shape) // 2 * 2 - 1)
        # O sinal de C depende da polaridade para que o fundo fique preto.
        correction = 5 if profile.polarity == "dark" else -5
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     kind, size, correction)
    else:
        threshold, mask = cv2.threshold(gray, 0, 255, kind | cv2.THRESH_OTSU)
        if profile.otsu_offset:
            threshold = max(0, min(255, threshold + profile.otsu_offset))
            _, mask = cv2.threshold(gray, threshold, 255, kind)
    return roi, mask


def prepare_digit(mask, profile, box):
    """Retifica segmentos inclinados usando a geometria informada no perfil."""
    cell = box.crop(mask)
    height, width = cell.shape[:2]
    if profile.digit_slant:
        shift = round(profile.digit_slant * width)
        matrix = np.array([[1, shift / max(1, height - 1), -max(shift, 0)],
                           [0, 1, 0]], dtype=np.float32)
        cell = cv2.warpAffine(cell, matrix, (width - abs(shift), height),
                              flags=cv2.INTER_NEAREST, borderValue=0)
    return cell


def activation(mask, box, threshold, margin):
    sample = box.crop(mask)
    occupancy = float(np.count_nonzero(sample)) / sample.size
    if abs(occupancy - threshold) < margin:
        raise ValueError("Ativação ambígua; ajuste iluminação, regiões ou limiares.")
    active = occupancy > threshold
    confidence = ((occupancy - threshold) / (1 - threshold) if active
                  else (threshold - occupancy) / threshold)
    return int(active), max(0.0, min(1.0, confidence))


def segment_sample(cell, box, search=0):
    """Busca o traço próximo à sonda sem mudar seu tamanho ou inferir o dígito."""
    if not search:
        return box
    height, width = cell.shape[:2]
    x0, y0, x1, y1 = box.pixels(width, height)
    sample_width, sample_height = x1 - x0, y1 - y0
    rx, ry = round(width * search), round(height * search)
    xs = np.arange(max(0, x0 - rx), min(width - sample_width, x0 + rx) + 1)
    ys = np.arange(max(0, y0 - ry), min(height - sample_height, y0 + ry) + 1)
    summed = cv2.integral((cell > 0).astype(np.uint8))
    counts = (summed[(ys + sample_height)[:, None], (xs + sample_width)[None, :]]
              - summed[ys[:, None], (xs + sample_width)[None, :]]
              - summed[(ys + sample_height)[:, None], xs[None, :]]
              + summed[ys[:, None], xs[None, :]])
    iy, ix = np.unravel_index(np.argmax(counts), counts.shape)
    return Box.from_pixels((int(xs[ix]), int(ys[iy]), sample_width, sample_height), width, height)


class DisplayReader:
    def __init__(self, profile, weights_path=None):
        self.profile = profile
        self.tesseract = None
        self.yolo = None
        if profile.engine == "yolo":
            from .yolo_digits import YoloDigits
            self.yolo = YoloDigits(profile, weights_path)
        if profile.engine == "tesseract":
            self.tesseract = shutil.which("tesseract")
            if not self.tesseract:
                raise ValueError("Instale o executável Tesseract OCR e adicione-o ao PATH.")

    def read(self, frame):
        raw, confidence = "", 0.0
        if self.yolo:
            self.yolo.last_detections = []
        try:
            if self.profile.engine == "yolo":
                raw, confidence = self.yolo.read(display_image(frame, self.profile))
            elif self.profile.engine == "seven_segment":
                _, mask = preprocess(frame, self.profile)
                raw, confidence = self._segments(mask)
            else:
                _, mask = preprocess(frame, self.profile)
                raw, confidence = self._ocr(mask)
            value = parse_number(raw)
            if self.profile.require_decimal and "." not in raw and "," not in raw:
                raise ValueError("O perfil exige um ponto decimal visível.")
            if confidence < self.profile.min_confidence:
                raise ValueError("Qualidade abaixo do mínimo configurado.")
            if self.profile.value_min is not None and value < self.profile.value_min:
                raise ValueError("Valor abaixo do limite do perfil.")
            if self.profile.value_max is not None and value > self.profile.value_max:
                raise ValueError("Valor acima do limite do perfil.")
            return Reading(raw, value, confidence, detections=tuple(self.yolo.last_detections) if self.yolo else ())
        except ValueError as exc:
            return Reading(raw, None, confidence, str(exc), tuple(self.yolo.last_detections) if self.yolo else ())

    def _segments(self, mask):
        profile = self.profile
        chars, scores = [], []
        for index, box in enumerate(profile.digits, start=1):
            cell = prepare_digit(mask, profile, box)
            if cell.shape[0] < 20 or cell.shape[1] < 10:
                raise ValueError(f"Dígito {index} sem resolução suficiente (mínimo 10 × 20 px).")
            probes = [activation(cell, segment_sample(cell, p, profile.segment_search),
                                 profile.segment_on, profile.ambiguity_margin)
                      for p in (profile.segment_probes or PROBES)]
            bits = tuple(item[0] for item in probes)
            if bits not in PATTERNS:
                raise ValueError(f"Padrão desconhecido no dígito {index}: {bits}.")
            chars.append(PATTERNS[bits])
            scores.extend(item[1] for item in probes)
        active_points = []
        if profile.decimal_mode == "detect":
            for point in profile.decimal_points:
                on, score = activation(mask, point.box, profile.point_on,
                                       profile.ambiguity_margin)
                scores.append(score)
                if on:
                    active_points.append(point.after)
            if len(active_points) > 1:
                raise ValueError("Vários pontos decimais ativos.")
            if active_points:
                chars.insert(active_points[0], ".")
        text = "".join(chars).strip()
        if profile.sign is not None:
            on, score = activation(mask, profile.sign, profile.point_on,
                                   profile.ambiguity_margin)
            scores.append(score)
            if on:
                text = "-" + text
        if profile.decimal_mode == "fixed" and profile.decimal_places:
            # Escala explicitamente fixa; nunca deduzida de um ponto ausente.
            if not re.fullmatch(r"-?[0-9]+", text):
                raise ValueError("Dígitos inválidos para escala decimal fixa.")
            negative = text.startswith("-")
            digits = text.lstrip("-").zfill(profile.decimal_places + 1)
            text = ("-" if negative else "") + digits[:-profile.decimal_places] + "." + digits[-profile.decimal_places:]
        return text, min(scores, default=0.0)

    def _ocr(self, mask):
        # Tesseract recebe texto escuro sobre fundo claro, com borda e altura útil.
        scale = max(1.0, 100 / mask.shape[0])
        image = cv2.resize(255 - mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        image = cv2.copyMakeBorder(image, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        ok, encoded = cv2.imencode(".png", image)
        if not ok:
            raise ValueError("Não foi possível preparar a imagem para OCR.")
        try:
            result = subprocess.run(
                [self.tesseract, "stdin", "stdout", "-l", "eng", "--psm", "7",
                 "-c", "load_system_dawg=0", "-c", "load_freq_dawg=0", "tsv"],
                input=encoded.tobytes(), capture_output=True, timeout=5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ValueError(f"Falha ao executar OCR: {exc}") from exc
        if result.returncode:
            raise ValueError("Tesseract falhou: " + result.stderr.decode(errors="replace").strip())
        words, scores = [], []
        for row in csv.DictReader(io.StringIO(result.stdout.decode("utf-8")), delimiter="\t"):
            if row.get("text", "").strip():
                words.append(row["text"].strip())
                scores.append(max(0.0, float(row["conf"])) / 100)
        # Preserva separação: "24 1" não pode virar silenciosamente "241".
        return " ".join(words), min(scores, default=0.0)


class StabilityGate:
    """Exige quadros consecutivos; uma falha invalida imediatamente a leitura."""

    def __init__(self, frames=3):
        if type(frames) is not int or frames < 1:
            raise ValueError("stable-frames precisa ser um inteiro positivo.")
        self.frames = frames
        self.count = 0
        self.key = None

    def update(self, reading):
        if not reading.valid:
            self.count, self.key = 0, None
            return "invalid"
        key = (reading.raw, reading.value)
        self.count = self.count + 1 if key == self.key else 1
        self.key = key
        return "ok" if self.count >= self.frames else "unstable"
