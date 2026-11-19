"""Detecção de caracteres e montagem estrita da leitura, sem sondas geométricas."""

from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median

CLASSES = tuple(str(i) for i in range(10)) + ("dot", "minus")


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self):
        x0, y0, x1, y1 = self.xyxy
        return (x0 + x1) / 2, (y0 + y1) / 2


def assemble(detections, expected_digits=None):
    """Não descarta candidatos de ponto/sinal nem corrige dígitos desconhecidos."""
    if not detections:
        raise ValueError("YOLO não detectou caracteres no visor.")
    for item in detections:
        if item.label not in CLASSES:
            raise ValueError(f"Classe YOLO desconhecida: {item.label}.")
        if not 0 <= item.confidence <= 1 or not math.isfinite(item.confidence):
            raise ValueError("Confiança de detecção inválida.")
        x0, y0, x1, y1 = item.xyxy
        if not all(math.isfinite(v) for v in item.xyxy) or x0 < 0 or y0 < 0 or x1 <= x0 or y1 <= y0:
            raise ValueError("Caixa de detecção inválida.")
    digits = sorted((d for d in detections if d.label.isdigit()), key=lambda d: d.center[0])
    if not digits:
        raise ValueError("YOLO não detectou dígitos.")
    if expected_digits is not None and len(digits) != expected_digits:
        raise ValueError(f"Leitura incompleta: detectados {len(digits)} dígitos; esperados {expected_digits}.")
    digit_height = median(d.xyxy[3] - d.xyxy[1] for d in digits)
    digit_width = median(d.xyxy[2] - d.xyxy[0] for d in digits)
    middle = median(d.center[1] for d in digits)
    for digit in digits:
        if abs(digit.center[1] - middle) > digit_height * .35:
            raise ValueError("Dígitos em linhas diferentes; recorte apenas uma leitura.")
        if not .45 * digit_height <= digit.xyxy[3] - digit.xyxy[1] <= 1.6 * digit_height:
            raise ValueError("Alturas de dígitos incompatíveis na mesma leitura.")
    for i, item in enumerate(detections):
        for other in detections[i + 1:]:
            a, b = item.xyxy, other.xyxy
            area = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            smallest = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
            if area / smallest > .25:
                raise ValueError("Detecções sobrepostas ou conflitantes; leitura rejeitada.")
    dots = [d for d in detections if d.label == "dot"]
    signs = [d for d in detections if d.label == "minus"]
    if len(dots) > 1 or len(signs) > 1:
        raise ValueError("Mais de um ponto decimal ou sinal detectado.")
    if signs:
        sign = signs[0]
        if (sign.center[0] >= digits[0].center[0] or abs(sign.center[1] - middle) > digit_height * .3
                or sign.xyxy[3] - sign.xyxy[1] > digit_height * .4):
            raise ValueError("Sinal negativo fora da posição esperada.")
    if dots:
        dot = dots[0]
        if (not digits[0].center[0] < dot.center[0] < digits[-1].center[0]
                or not middle + .15 * digit_height <= dot.center[1] <= middle + .7 * digit_height
                or dot.xyxy[3] - dot.xyxy[1] > digit_height * .4
                or dot.xyxy[2] - dot.xyxy[0] > max(digit_width * .6, digit_height * .2)):
            raise ValueError("Ponto decimal fora da região esperada.")
    tokens = sorted(detections, key=lambda d: d.center[0])
    text = "".join({"dot": ".", "minus": "-"}.get(d.label, d.label) for d in tokens)
    return text, min(d.confidence for d in detections)


def ultralytics_yolo():
    import os
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".cache/ultralytics").resolve()))
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
    try:
        from ultralytics import YOLO, settings
    except ImportError as exc:
        raise ValueError("Instale o suporte YOLO: python -m pip install -r requirements-yolo.txt") from exc
    settings.update({key: False for key in ("sync", "hub", "wandb", "mlflow", "comet", "clearml", "neptune")
                     if key in settings})
    return YOLO


class YoloDigits:
    def __init__(self, profile, weights_path=None):
        self.profile = profile
        weights = Path(weights_path or profile.yolo_weights)
        if not weights.is_file():
            raise ValueError(f"Pesos de dígitos ainda não encontrados: {weights}. Colete, anote e treine com yolo-train.")
        self.model = ultralytics_yolo()(str(weights), task="detect")
        names = self.model.names
        self.names = dict(enumerate(names)) if isinstance(names, list) else names
        if sorted(self.names.values()) != sorted(CLASSES):
            raise ValueError("Pesos incompatíveis: o modelo deve ter as classes 0–9, dot e minus. Pesos COCO não leem o visor.")
        self.last_detections = []

    def detect(self, roi):
        self.last_detections = []
        try:
            result = self.model.predict(source=roi, imgsz=self.profile.yolo_imgsz,
                                        conf=self.profile.yolo_detection_conf, iou=.5,
                                        device=self.profile.yolo_device, max_det=32, verbose=False)[0]
        except Exception as exc:
            raise ValueError(f"Falha na inferência YOLO: {exc}") from exc
        detections = []
        for row in result.boxes.data.cpu().numpy():
            x0, y0, x1, y1, confidence, class_id = row[:6]
            if x1 > roi.shape[1] + 1 or y1 > roi.shape[0] + 1:
                raise ValueError("YOLO retornou caixa fora do recorte.")
            detections.append(Detection(self.names[int(class_id)], float(confidence),
                                        tuple(float(v) for v in (x0, y0, x1, y1))))
        self.last_detections = detections
        return detections

    def read(self, roi):
        return assemble(self.detect(roi), self.profile.yolo_expected_digits)
