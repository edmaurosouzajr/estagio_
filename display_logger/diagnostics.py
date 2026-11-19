"""Imagens e medições para verificar o enquadramento e os segmentos."""

from dataclasses import asdict
import json
from pathlib import Path
import textwrap
import unicodedata

import cv2
import numpy as np

from .vision import DisplayReader, PATTERNS, PROBES, prepare_digit, preprocess, segment_sample
from .geometry import display_image, rectify, roi_polygon


def write_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Não foi possível salvar a imagem {path}.")


def camera_overlay(frame, profile, color=(0, 100, 255)):
    annotated = frame.copy()
    polygon = np.rint(roi_polygon(frame, profile)).astype(np.int32)
    cv2.polylines(annotated, [polygon], True, color, 2)
    return annotated


def detection_overlay(cropped, detections, max_width=800, max_height=400):
    scale = min(max_width / cropped.shape[1], max_height / cropped.shape[0])
    output = cv2.resize(cropped, None, fx=scale, fy=scale)
    for detection in detections:
        x0, y0, x1, y1 = [round(v * scale) for v in detection.xyxy]
        cv2.rectangle(output, (x0, y0), (x1, y1), (0, 220, 0), 1)
        label = f"{detection.label} {detection.confidence:.2f}"
        cv2.putText(output, label, (x0, max(12, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 220, 0), 1)
    return output


def save_diagnostics(frame, profile, directory, reading=None, source_frame=None):
    directory = Path(directory)
    reading = reading if reading is not None else DisplayReader(profile).read(frame)
    write_image(directory / "original.png", source_frame if source_frame is not None else frame)
    if source_frame is not None:
        write_image(directory / "alinhado.png", frame)
    write_image(directory / "enquadramento.png", camera_overlay(frame, profile))
    plane = rectify(frame, profile.perspective)
    cropped = profile.roi.crop(plane)
    write_image(directory / "display.png", cropped)
    if profile.perspective:
        write_image(directory / "retificado.png", plane)
    report = {
        "frame_size": {"width": frame.shape[1], "height": frame.shape[0]},
        "roi_pixels": profile.roi.pixels(plane.shape[1], plane.shape[0]),
        "roi_space": "rectified" if profile.perspective else "camera",
        "perspective": asdict(profile.perspective) if profile.perspective else None,
        "reading": asdict(reading), "digits": [],
    }
    if profile.engine == "yolo":
        write_image(directory / "yolo-deteccoes.png", detection_overlay(cropped, reading.detections))
        (directory / "diagnostico.json").write_text(
            json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        return directory
    try:
        _, mask = preprocess(frame, profile)
        write_image(directory / "mascara.png", mask)
        for index, box in enumerate(profile.digits, 1):
            cell = prepare_digit(mask, profile, box)
            occupancies = []
            annotated = cv2.cvtColor(cell, cv2.COLOR_GRAY2BGR)
            for probe in (profile.segment_probes or PROBES):
                probe = segment_sample(cell, probe, profile.segment_search)
                sample = probe.crop(cell)
                occupancy = float(np.count_nonzero(sample) / sample.size)
                occupancies.append(occupancy)
                x0, y0, x1, y1 = probe.pixels(cell.shape[1], cell.shape[0])
                color = (0, 200, 0) if occupancy > profile.segment_on else (0, 0, 220)
                cv2.rectangle(annotated, (x0, y0), (x1 - 1, y1 - 1), color, 1)
            bits = tuple(int(value > profile.segment_on) for value in occupancies)
            report["digits"].append({
                "position": index, "size": {"width": cell.shape[1], "height": cell.shape[0]},
                "segment_occupancy": dict(zip("abcdefg", occupancies)),
                "pattern": bits, "character": PATTERNS.get(bits, "?"),
            })
            write_image(directory / f"digito-{index}.png", box.crop(cropped))
            write_image(directory / f"digito-{index}-sondas.png",
                        cv2.resize(annotated, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST))
    except ValueError as exc:
        report["preprocessing_error"] = str(exc)
    (directory / "diagnostico.json").write_text(
        json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    return directory


def preview_image(frame, profile, reading, status):
    """Mostra o quadro completo ao lado do recorte para evidenciar deslocamentos."""
    color = (60, 240, 60) if status == "ok" else (80, 150, 255)
    canvas = np.full((470, 1120, 3), 30, dtype=np.uint8)
    camera = camera_overlay(frame, profile, color)
    scale = min(640 / camera.shape[1], 350 / camera.shape[0])
    camera = cv2.resize(camera, None, fx=scale, fy=scale)
    canvas[90:90 + camera.shape[0], 10:10 + camera.shape[1]] = camera
    cropped = display_image(frame, profile).copy()
    height, width = cropped.shape[:2]
    for index, box in enumerate(profile.digits, 1):
        x0, y0, x1, y1 = box.pixels(width, height)
        cv2.rectangle(cropped, (x0, y0), (x1 - 1, y1 - 1), (255, 180, 0), 1)
    for box in [p.box for p in profile.decimal_points] + ([profile.sign] if profile.sign else []):
        x0, y0, x1, y1 = box.pixels(width, height)
        cv2.rectangle(cropped, (x0, y0), (x1 - 1, y1 - 1), (0, 180, 255), 1)
    if profile.engine == "yolo":
        cropped = detection_overlay(cropped, reading.detections, 450, 250)
    else:
        scale = min(450 / width, 250 / height)
        cropped = cv2.resize(cropped, None, fx=scale, fy=scale)
    canvas[90:90 + cropped.shape[0], 660:660 + cropped.shape[1]] = cropped
    label = f"{status}: {reading.raw or '?'} {profile.unit} | Q: sair | S: salvar diagnostico"
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, color, 1)
    cv2.putText(canvas, "Camera: confira o contorno selecionado", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, .5, (230,) * 3, 1)
    cv2.putText(canvas, "Display ampliado", (660, 72), cv2.FONT_HERSHEY_SIMPLEX, .5, (230,) * 3, 1)
    if reading.reason:
        reason = unicodedata.normalize("NFKD", reading.reason).encode("ascii", "ignore").decode()
        for index, line in enumerate(textwrap.wrap(reason, width=57)[:4]):
            cv2.putText(canvas, line, (660, 370 + 22 * index), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
    return canvas
