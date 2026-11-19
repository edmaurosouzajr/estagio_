"""Treinamento supervisionado e avaliação da leitura completa em sessões separadas."""

from collections import Counter
import json
from pathlib import Path
import shutil

import cv2

from .config import Box, Profile
from .vision import DisplayReader
from .yolo_data import digest, read_labels
from .yolo_digits import CLASSES, ultralytics_yolo


def checked_dataset(data_path):
    data_path = Path(data_path).resolve()
    try:
        config = json.loads(data_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("Use o data.yaml gerado por yolo-prepare, acompanhado do manifest.json.") from exc
    root = data_path.parent
    if list(config.get("names", [])) != list(CLASSES) or Path(config.get("path", "")).resolve() != root:
        raise ValueError("Classes ou caminho do dataset incompatíveis; gere novamente com yolo-prepare.")
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sessions, hashes, counts = {}, {}, Counter()
    for sample in manifest.get("images", []):
        relative = Path(sample["image"])
        split = sample["split"]
        if split not in ("train", "val", "test") or relative.parts[:2] != ("images", split) or len(relative.parts) != 3:
            raise ValueError("Caminho ou split inválido no manifesto.")
        image = root / relative
        label = root / "labels" / split / (image.stem + ".txt")
        if digest(image) != sample["sha256"] or digest(label) != sample.get("labels_sha256"):
            raise ValueError(f"Amostra mudou depois da preparação: {image}.")
        if sessions.setdefault(sample["session"], split) != split or hashes.setdefault(sample["sha256"], split) != split:
            raise ValueError("Há vazamento entre treino, validação e teste.")
        labels = read_labels(label)
        if split == "train":
            counts.update(CLASSES[c] for c, *_ in labels)
    if set(sessions.values()) != {"train", "val", "test"}:
        raise ValueError("Dataset precisa conter treino, validação e teste independentes.")
    manifest["missing_train_classes"] = [c for c in CLASSES if not counts[c]]
    return root, manifest


def train_model(args):
    data, manifest = checked_dataset(args.data)
    if manifest["missing_train_classes"]:
        raise ValueError("Faltam exemplos reais revisados no treino: " + ", ".join(manifest["missing_train_classes"]))
    output = Path(args.output).resolve()
    if output.exists():
        raise ValueError(f"{output} já existe; escolha outro --output para preservar o modelo anterior.")
    if args.imgsz % 32 or not 128 <= args.imgsz <= 2048:
        raise ValueError("--imgsz precisa ser múltiplo de 32 entre 128 e 2048.")
    model = ultralytics_yolo()(args.model, task="detect")
    try:
        model.train(data=str(Path(args.data).resolve()), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, project=str(Path(args.project).resolve()),
                    name="train", exist_ok=False, workers=0, seed=42,
                    fliplr=0., flipud=0., mosaic=0., mixup=0., degrees=5.,
                    translate=.05, scale=.15, hsv_h=0., hsv_s=.1, hsv_v=.3)
    except Exception as exc:
        raise ValueError(f"Treinamento interrompido: {exc}") from exc
    best = Path(model.trainer.best)
    if not best.is_file():
        raise ValueError("O treinamento não produziu best.pt.")
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, output)
    output.with_suffix(".training.json").write_text(json.dumps({
        "dataset": str(data), "sessions": manifest["sessions"], "classes": list(CLASSES),
        "epochs_requested": args.epochs, "source_checkpoint": str(best),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def evaluate_model(args):
    root, manifest = checked_dataset(args.data)
    profile = Profile(name="avaliacao-yolo", unit="unidade-do-teste", roi=Box(0, 0, 1, 1), engine="yolo",
                      yolo_weights=str(Path(args.weights).resolve()), yolo_imgsz=args.imgsz,
                      yolo_device=args.device, yolo_expected_digits=args.expected_digits,
                      require_decimal=not args.allow_integer)
    reader = DisplayReader(profile)
    rows = []
    for sample in manifest["images"]:
        if sample["split"] != "test":
            continue
        image = cv2.imread(str(root / sample["image"]))
        if image is None:
            raise ValueError(f"Imagem de teste inválida: {sample['image']}.")
        reading = reader.read(image)
        expected = sample["text"]
        correct = (reading.valid and reading.raw == expected) if expected is not None else not reading.valid
        rows.append({"image": sample["image"], "expected": expected, "predicted": reading.raw,
                     "accepted": reading.valid, "correct": correct, "confidence": reading.confidence,
                     "reason": reading.reason})
    positives = [r for r in rows if r["expected"] is not None]
    negatives = [r for r in rows if r["expected"] is None]
    result = {
        "test_images": len(rows), "positive_images": len(positives), "negative_images": len(negatives),
        "correct_complete_readings": sum(r["correct"] for r in positives),
        "complete_reading_accuracy": sum(r["correct"] for r in positives) / len(positives) if positives else None,
        "incorrect_accepted_readings": sum(r["accepted"] and not r["correct"] for r in positives),
        "negative_false_accepts": sum(r["accepted"] for r in negatives),
        "rejected_positive_readings": sum(not r["accepted"] for r in positives),
        "images": rows,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in result.items() if k != "images"}
