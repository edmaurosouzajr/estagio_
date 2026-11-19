"""Perfis usam coordenadas normalizadas: x, y, largura, altura entre 0 e 1."""

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    w: float
    h: float

    def __post_init__(self):
        values = (self.x, self.y, self.w, self.h)
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) for v in values):
            raise ValueError("Coordenadas precisam ser números finitos.")
        if (self.x < 0 or self.y < 0 or self.w <= 0 or self.h <= 0
                or self.x + self.w > 1.000001 or self.y + self.h > 1.000001):
            raise ValueError("Região fora da imagem; use x, y, w, h normalizados.")

    @classmethod
    def from_pixels(cls, rect, width, height):
        x, y, w, h = rect
        return cls(x / width, y / height, w / width, h / height)

    def pixels(self, width, height):
        x0, y0 = round(self.x * width), round(self.y * height)
        x1 = min(width, round((self.x + self.w) * width))
        y1 = min(height, round((self.y + self.h) * height))
        if x1 <= x0 or y1 <= y0:
            raise ValueError("Região pequena demais para a resolução da imagem.")
        return x0, y0, x1, y1

    def crop(self, image):
        x0, y0, x1, y1 = self.pixels(image.shape[1], image.shape[0])
        return image[y0:y1, x0:x1]


@dataclass(frozen=True)
class DecimalPoint:
    after: int  # Posição de dígito, a partir de 1; inclui posições em branco.
    box: Box


@dataclass(frozen=True)
class Perspective:
    # Superior esquerdo, superior direito, inferior direito, inferior esquerdo.
    corners: list[tuple[float, float]]
    width: int
    height: int

    def __post_init__(self):
        if len(self.corners) != 4:
            raise ValueError("Selecione exatamente quatro cantos do visor.")
        for point in self.corners:
            if len(point) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not math.isfinite(v) or not 0 <= v <= 1 for v in point):
                raise ValueError("Os cantos precisam ter coordenadas normalizadas entre 0 e 1.")
        for i in range(4):
            a, b, c = [self.corners[j % 4] for j in (i, i + 1, i + 2)]
            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if cross <= 1e-8:
                raise ValueError("Cantos inválidos: siga SE, SD, ID, IE, sem cruzar as linhas.")
        if any(type(v) is not int or not 8 <= v <= 4096 for v in (self.width, self.height)):
            raise ValueError("O visor retificado precisa ter largura e altura entre 8 e 4096 pixels.")


@dataclass
class Profile:
    name: str
    unit: str
    roi: Box
    engine: str = "seven_segment"
    digits: list[Box] = field(default_factory=list)
    decimal_mode: str = "detect"  # detect, fixed ou none
    decimal_points: list[DecimalPoint] = field(default_factory=list)
    decimal_places: int = 0
    require_decimal: bool = False
    sign: Box | None = None
    polarity: str = "dark"
    threshold: str = "otsu"
    otsu_offset: float = 0.0
    digit_slant: float = 0.0
    segment_probes: list[Box] = field(default_factory=list)
    segment_search: float = 0.0
    track_display: bool = False
    reference_image: str | None = None
    perspective: Perspective | None = None
    segment_on: float = 0.45
    point_on: float = 0.30
    ambiguity_margin: float = 0.08
    min_confidence: float = 0.55
    value_min: float | None = None
    value_max: float | None = None
    yolo_weights: str | None = None
    yolo_imgsz: int = 640
    yolo_device: str = "cpu"
    yolo_detection_conf: float = .15
    yolo_expected_digits: int | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("O perfil precisa de um nome.")
        if not isinstance(self.unit, str) or not self.unit.strip():
            raise ValueError("O perfil precisa de uma unidade.")
        for value, options, label in (
            (self.engine, ("seven_segment", "tesseract", "yolo"), "engine"),
            (self.polarity, ("dark", "light"), "polarity"),
            (self.threshold, ("otsu", "adaptive"), "threshold"),
            (self.decimal_mode, ("detect", "fixed", "none"), "decimal_mode"),
        ):
            if value not in options:
                raise ValueError(f"{label} precisa ser um de {options}.")
        for attr in ("segment_on", "point_on", "ambiguity_margin", "min_confidence"):
            value = getattr(self, attr)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not 0 < value < 1):
                raise ValueError(f"{attr} precisa estar entre 0 e 1, sem os extremos.")
        for attr, low, high in (("otsu_offset", -64, 64), ("digit_slant", -.5, .5)):
            value = getattr(self, attr)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not low <= value <= high):
                raise ValueError(f"{attr} precisa estar entre {low} e {high}.")
        if (isinstance(self.segment_search, bool) or not isinstance(self.segment_search, (int, float))
                or not math.isfinite(self.segment_search) or not 0 <= self.segment_search <= .10):
            raise ValueError("segment_search precisa estar entre 0 e 0.10 da célula.")
        if self.threshold != "otsu" and self.otsu_offset:
            raise ValueError("otsu_offset só é aplicado ao limiar Otsu.")
        if self.segment_probes and len(self.segment_probes) != 7:
            raise ValueError("segment_probes precisa conter sete regiões na ordem a, b, c, d, e, f, g.")
        if not isinstance(self.track_display, bool):
            raise ValueError("track_display precisa ser booleano.")
        if self.reference_image is not None and (
                not isinstance(self.reference_image, str) or not self.reference_image.strip()):
            raise ValueError("reference_image precisa ser um caminho preenchido.")
        if self.track_display and not self.reference_image:
            raise ValueError("O rastreamento exige reference_image.")
        if type(self.yolo_imgsz) is not int or not 128 <= self.yolo_imgsz <= 2048 or self.yolo_imgsz % 32:
            raise ValueError("yolo_imgsz deve ser múltiplo de 32, entre 128 e 2048.")
        if not isinstance(self.yolo_device, str) or not self.yolo_device.strip():
            raise ValueError("Informe yolo_device, por exemplo cpu ou 0.")
        if (isinstance(self.yolo_detection_conf, bool) or not isinstance(self.yolo_detection_conf, (int, float))
                or not math.isfinite(self.yolo_detection_conf) or not 0 < self.yolo_detection_conf < 1):
            raise ValueError("yolo_detection_conf deve estar entre 0 e 1.")
        if self.yolo_expected_digits is not None and (type(self.yolo_expected_digits) is not int
                                                     or self.yolo_expected_digits < 1):
            raise ValueError("yolo_expected_digits deve ser inteiro positivo ou null.")
        if self.engine == "yolo" and (not isinstance(self.yolo_weights, str) or not self.yolo_weights.strip()):
            raise ValueError("O motor YOLO exige o caminho yolo_weights para pesos treinados.")
        if self.ambiguity_margin >= min(self.segment_on, 1 - self.segment_on,
                                        self.point_on, 1 - self.point_on):
            raise ValueError("ambiguity_margin excede os limiares de ativação.")
        if (isinstance(self.decimal_places, bool) or not isinstance(self.decimal_places, int)
                or not 0 <= self.decimal_places <= 12):
            raise ValueError("decimal_places precisa ser um inteiro entre 0 e 12.")
        if not isinstance(self.require_decimal, bool):
            raise ValueError("require_decimal precisa ser booleano.")
        if self.require_decimal and (self.decimal_mode == "none" or (
                self.decimal_mode == "fixed" and self.decimal_places == 0)):
            raise ValueError("require_decimal exige separador detectado ou escala fixa com casas decimais.")
        if self.engine == "seven_segment":
            if not self.digits:
                raise ValueError("Informe ao menos uma região de dígito.")
            if any(a.x >= b.x for a, b in zip(self.digits, self.digits[1:])):
                raise ValueError("Os dígitos precisam estar em ordem da esquerda para a direita.")
            positions = [p.after for p in self.decimal_points]
            if any(type(p) is not int or not 1 <= p <= len(self.digits) for p in positions):
                raise ValueError("Posição de ponto decimal fora dos dígitos.")
            if len(set(positions)) != len(positions):
                raise ValueError("Existem posições de ponto decimal repetidas.")
            if self.decimal_mode == "detect" and not self.decimal_points:
                raise ValueError("Modo detect exige regiões de ponto; use fixed ou none.")
        elif self.decimal_mode != "detect":
            raise ValueError("No OCR, o separador decimal é sempre lido da imagem (detect).")
        for value in (self.value_min, self.value_max):
            if value is not None and (isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError("Os limites de valor precisam ser números finitos.")
        if (self.value_min is not None and self.value_max is not None
                and self.value_min > self.value_max):
            raise ValueError("value_min não pode superar value_max.")

    def save(self, path):
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("O perfil precisa ser um objeto JSON.")
        try:
            data["roi"] = Box(**data["roi"])
            data["digits"] = [Box(**b) for b in data.get("digits", [])]
            data["segment_probes"] = [Box(**b) for b in data.get("segment_probes", [])]
            data["decimal_points"] = [DecimalPoint(p["after"], Box(**p["box"]))
                                      for p in data.get("decimal_points", [])]
            if data.get("sign") is not None:
                data["sign"] = Box(**data["sign"])
            if data.get("perspective") is not None:
                spec = data["perspective"]
                data["perspective"] = Perspective(
                    [tuple(point) for point in spec["corners"]], spec["width"], spec["height"])
            return cls(**data)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Estrutura inválida do perfil: {exc}") from exc
