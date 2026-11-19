"""Imagens revisadas por sessão, rótulos YOLO e separação sem misturar sessões."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import tempfile
import time
import uuid

import cv2

from .yolo_digits import CLASSES, Detection, assemble


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session_dir(root, session):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", session):
        raise ValueError("Sessão: use até 64 letras, números, hífen ou sublinhado, começando por letra/número.")
    return Path(root) / "raw" / session


def save_sample(root, session, image, metadata=None):
    folder = session_dir(root, session)
    images = folder / "images"
    images.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex[:16]
    target = images / f"{name}.png"
    if not cv2.imwrite(str(target), image):
        raise OSError(f"Falha ao salvar amostra {target}.")
    info = {"timestamp": time.time(), "session": session, "width": image.shape[1],
            "height": image.shape[0], "sha256": digest(target), **(metadata or {})}
    target.with_suffix(".source.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def labels_path(image):
    return image.parent.parent / "labels" / (image.stem + ".txt")


def review_path(image):
    return image.parent.parent / "labels" / (image.stem + ".review.json")


def read_labels(path):
    labels = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        try:
            if len(parts) != 5:
                raise ValueError("esperados classe, centro_x, centro_y, largura, altura")
            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
            if not 0 <= class_id < len(CLASSES) or not (0 < w <= 1 and 0 < h <= 1
                    and x - w / 2 >= -1e-6 and y - h / 2 >= -1e-6
                    and x + w / 2 <= 1.000001 and y + h / 2 <= 1.000001):
                raise ValueError("classe ou coordenadas fora dos limites")
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"Rótulo inválido em {path}, linha {number}: {exc}") from exc
        labels.append((class_id, x, y, w, h))
    return labels


def label_detections(labels, width, height):
    return [Detection(CLASSES[c], 1., (max(0., (x - w / 2) * width), max(0., (y - h / 2) * height),
                                      (x + w / 2) * width, (y + h / 2) * height))
            for c, x, y, w, h in labels]


def save_review(image_path, labels, negative=False):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Imagem inválida: {image_path}.")
    if negative and labels:
        raise ValueError("Uma imagem negativa deve ter rótulos vazios.")
    text = None if negative else assemble(label_detections(labels, image.shape[1], image.shape[0]))[0]
    target = labels_path(image_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(f"{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n" for c, x, y, w, h in labels)
    # Valida o texto serializado antes de marcar a revisão como concluída.
    temporary = target.with_suffix(".tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        read_labels(temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    review = {"text": text, "negative": negative, "image_sha256": digest(image_path),
              "labels_sha256": digest(target), "reviewed_at": time.time()}
    review_path(image_path).write_text(json.dumps(review, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return text


def reviewed_samples(root):
    samples, pending = [], 0
    for image in sorted((Path(root) / "raw").glob("*/images/*.png")):
        label, review_file = labels_path(image), review_path(image)
        if not label.exists() or not review_file.exists():
            pending += 1
            continue
        review = json.loads(review_file.read_text(encoding="utf-8"))
        if review.get("image_sha256") != digest(image) or review.get("labels_sha256") != digest(label):
            raise ValueError(f"Imagem ou rótulo mudou após a revisão: {image}. Revise novamente.")
        labels = read_labels(label)
        matrix = cv2.imread(str(image))
        if matrix is None:
            raise ValueError(f"Imagem inválida: {image}.")
        expected = None if review.get("negative") else assemble(
            label_detections(labels, matrix.shape[1], matrix.shape[0]))[0]
        if expected != review.get("text") or (review.get("negative") and labels):
            raise ValueError(f"Revisão inconsistente: {review_file}.")
        samples.append({"image": image, "label": label, "session": image.parent.parent.name,
                        "text": expected, "sha256": review["image_sha256"], "labels": labels})
    return samples, pending


def dataset_status(root):
    samples, pending = reviewed_samples(root)
    counts = Counter(CLASSES[c] for sample in samples for c, *_ in sample["labels"])
    return {"reviewed_images": len(samples), "pending_images": pending,
            "sessions": dict(Counter(s["session"] for s in samples)),
            "classes": dict(counts), "missing_classes": [c for c in CLASSES if not counts[c]],
            "negative_images": sum(s["text"] is None for s in samples)}


def prepare_dataset(root, output, val_session=None, test_session=None, seed=42):
    samples, pending = reviewed_samples(root)
    groups = sorted({s["session"] for s in samples})
    if len(groups) < 3:
        raise ValueError("São necessárias pelo menos 3 sessões revisadas: treino, validação e teste independentes.")
    if bool(val_session) != bool(test_session):
        raise ValueError("Informe --val-session e --test-session juntos.")
    if val_session is None:
        random.Random(seed).shuffle(groups)
        val_session, test_session = groups[:2]
    if val_session == test_session or val_session not in groups or test_session not in groups:
        raise ValueError("Validação e teste devem ser sessões existentes e diferentes.")
    assignments = {group: "val" if group == val_session else "test" if group == test_session else "train"
                   for group in groups}
    hashes = {}
    for sample in samples:
        split = assignments[sample["session"]]
        previous = hashes.setdefault(sample["sha256"], split)
        if previous != split:
            raise ValueError("Existem imagens idênticas em sessões de splits diferentes; remova a duplicação na coleta.")
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f"A saída {output} já existe. Escolha outra pasta para não misturar versões do dataset.")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".digits-", dir=output.parent))
    try:
        records, train_counts = [], Counter()
        for sample in samples:
            split = assignments[sample["session"]]
            name = sample["session"] + "__" + sample["image"].stem
            for kind, source, suffix in (("images", sample["image"], ".png"), ("labels", sample["label"], ".txt")):
                target = temporary / kind / split / (name + suffix)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            if split == "train":
                train_counts.update(CLASSES[c] for c, *_ in sample["labels"])
            records.append({"image": f"images/{split}/{name}.png", "session": sample["session"],
                            "split": split, "text": sample["text"], "sha256": sample["sha256"],
                            "labels_sha256": digest(sample["label"])})
        data = {"path": str(output), "train": "images/train", "val": "images/val", "test": "images/test",
                "names": list(CLASSES)}
        (temporary / "data.yaml").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        manifest = {"sessions": assignments, "images": records, "pending_skipped": pending,
                    "train_classes": dict(train_counts), "missing_train_classes": [c for c in CLASSES if not train_counts[c]]}
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.rename(output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
