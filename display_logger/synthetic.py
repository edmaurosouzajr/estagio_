"""Cenas artificiais para demonstração; não representam medições de bancada."""

from pathlib import Path
import time

import cv2
import numpy as np

from .config import Box, DecimalPoint, Profile
from .records import JsonlWriter, make_record, to_csv
from .vision import DisplayReader, StabilityGate

# Polígonos dos segmentos, em uma célula de 50 × 100; independentes das sondas.
SHAPES = (
    [(7, 2), (43, 2), (48, 7), (42, 12), (8, 12), (2, 7)],
    [(43, 10), (49, 15), (49, 43), (43, 49), (39, 43), (39, 16)],
    [(43, 51), (49, 57), (49, 85), (43, 91), (39, 85), (39, 57)],
    [(8, 88), (42, 88), (48, 93), (43, 98), (7, 98), (2, 93)],
    [(7, 51), (11, 57), (11, 85), (7, 91), (1, 85), (1, 57)],
    [(7, 10), (11, 16), (11, 43), (7, 49), (1, 43), (1, 15)],
    [(8, 45), (42, 45), (48, 50), (42, 55), (8, 55), (2, 50)],
)
# Segmentos ativos escritos separadamente do dicionário do reconhecedor.
SCENE_SEGMENTS = {
    "0": (0, 1, 2, 3, 4, 5), "1": (1, 2), "2": (0, 1, 3, 4, 6),
    "3": (0, 1, 2, 3, 6), "4": (1, 2, 5, 6), "5": (0, 2, 3, 5, 6),
    "6": (0, 2, 3, 4, 5, 6), "7": (0, 1, 2), "8": tuple(range(7)),
    "9": (0, 1, 2, 3, 5, 6), "-": (6,), " ": (), "O": (0, 1, 2, 3, 4, 5),
    "L": (3, 4, 5), "?": (0, 1),
}


def scene(text, count=4, polarity="dark", decimal_mode="detect", decimal_places=0):
    negative = text.startswith("-")
    unsigned = text.lstrip("-")
    dot = unsigned.find(".")
    chars = unsigned.replace(".", "")
    if len(chars) > count:
        raise ValueError("Texto maior do que o display sintético.")
    padding = count - len(chars)
    chars = chars.rjust(count)
    point_after = padding + dot if dot >= 0 else None
    width, height = count * 82 + 70, 150
    background, ink = ((220, 25) if polarity == "dark" else (20, 245))
    image = np.full((height, width, 3), background, dtype=np.uint8)
    digits, points = [], []
    for index, char in enumerate(chars):
        x, y = 50 + index * 82, 25
        for segment in SCENE_SEGMENTS[char]:
            polygon = np.array(SHAPES[segment], dtype=np.int32) + (x, y)
            cv2.fillConvexPoly(image, polygon, (ink,) * 3)
        digits.append(Box.from_pixels((x, y, 50, 100), width, height))
        if index < count - 1:
            point_rect = (x + 58, y + 86, 10, 10)
            points.append(DecimalPoint(index + 1, Box.from_pixels(point_rect, width, height)))
            if index + 1 == point_after:
                cv2.circle(image, (x + 63, y + 91), 5, (ink,) * 3, -1)
    if negative:
        cv2.rectangle(image, (13, 71), (33, 79), (ink,) * 3, -1)
    profile = Profile(
        name="demonstracao-sintetica", unit="volts", roi=Box(0, 0, 1, 1),
        digits=digits, decimal_points=points if count > 1 else [],
        sign=Box.from_pixels((12, 70, 23, 11), width, height), polarity=polarity,
        decimal_mode=decimal_mode, decimal_places=decimal_places,
    )
    return image, profile


def create_demo(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    _, profile = scene("24.1")
    profile.save(directory / "perfil.json")
    reader, gate = DisplayReader(profile), StabilityGate(3)
    base_time = time.time()
    log_path = directory / "leituras.txt"
    # O log é append-only, inclusive quando o demo é executado novamente.
    with JsonlWriter(log_path) as log:
        for index, text in enumerate(("24.1", "-0.08", "120.0", "12.34", "OL", "")):
            image, _ = scene(text)
            cv2.imwrite(str(directory / f"display-{index + 1}.png"), image)
            reading = reader.read(image)
            for _ in range(3):
                status = gate.update(reading)
            log.write(make_record(base_time + index, "DEMO-SINTETICO", profile, reading, status))
    count = to_csv(log_path, directory / "leituras.csv")
    return count
