"""TXT com um objeto JSON por linha (JSONL) e conversor CSV transacional."""

import csv
import json
import math
import os
from pathlib import Path
import tempfile

FIELDS = ("timestamp", "id-teste", "unidade medida", "value", "status",
          "confidence", "raw", "reason", "profile")


def make_record(timestamp, test_id, profile, reading, status):
    return {
        "timestamp": timestamp,
        "id-teste": test_id,
        "unidade medida": profile.unit,
        "value": reading.value if status == "ok" else None,
        "status": status,
        "confidence": round(reading.confidence, 4),
        "raw": reading.raw,
        "reason": reading.reason if status == "invalid" else (
            "Aguardando quadros consecutivos iguais." if status == "unstable" else ""),
        "profile": profile.name,
    }


class JsonlWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.handle = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and self.path.stat().st_size:
            with self.path.open("rb") as existing:
                existing.seek(-1, 2)
                if existing.read(1) != b"\n":
                    raise ValueError("Log existente não termina com quebra de linha; use outro arquivo.")
        self.handle = self.path.open("a", encoding="utf-8")
        return self

    def write(self, record):
        self.handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        self.handle.flush()

    def __exit__(self, *args):
        self.handle.close()


def _finite_number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{label} precisa ser numérico.")
    try:
        number = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{label} precisa ser numérico.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} precisa ser finito.")


def validate_record(record):
    if not isinstance(record, dict):
        raise ValueError("A linha precisa conter um objeto JSON.")
    for key in FIELDS[:4]:
        if key not in record:
            raise ValueError(f"Campo obrigatório ausente: {key}.")
    _finite_number(record["timestamp"], "timestamp")
    for key in ("id-teste", "unidade medida"):
        if not isinstance(record[key], str) or not record[key].strip():
            raise ValueError(f"{key} precisa ser um texto preenchido.")
    status = record.get("status", "ok")
    if status not in ("ok", "invalid", "unstable"):
        raise ValueError("status desconhecido.")
    if record["value"] is not None:
        _finite_number(record["value"], "value")
    if status == "ok" and record["value"] is None:
        raise ValueError("Registro ok precisa conter value.")
    if status != "ok" and record["value"] is not None:
        raise ValueError("Registros inválidos ou instáveis precisam ter value nulo.")
    return {**record, "status": status}


def to_csv(source, target, delimiter=","):
    source, target = Path(source), Path(target)
    if source.resolve() == target.resolve():
        raise ValueError("O CSV de saída não pode ser o próprio TXT de entrada.")
    if len(delimiter) != 1 or delimiter in "\r\n\0":
        raise ValueError("O delimitador precisa ser um único caractere válido.")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    count = 0
    try:
        with source.open(encoding="utf-8-sig") as incoming, tempfile.NamedTemporaryFile(
            "w", encoding="utf-8-sig", newline="", dir=target.parent,
            prefix=".display-csv-", suffix=".tmp", delete=False,
        ) as outgoing:
            temporary = Path(outgoing.name)
            writer = csv.DictWriter(outgoing, fieldnames=FIELDS, extrasaction="ignore", delimiter=delimiter)
            writer.writeheader()
            for line_number, line in enumerate(incoming, start=1):
                if not line.strip():
                    continue
                try:
                    record = validate_record(json.loads(line))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"Linha {line_number}: {exc}") from exc
                writer.writerow(record)
                count += 1
        os.replace(temporary, target)
        temporary = None
        return count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
