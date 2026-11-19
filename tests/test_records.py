import csv
import json

import pytest

from display_logger.records import JsonlWriter, make_record, to_csv
from display_logger.synthetic import scene
from display_logger.vision import Reading


def test_log_csv_roundtrip_with_accents_and_unstable_reading(tmp_path):
    _, profile = scene("24.1")
    source, target = tmp_path / "teste.txt", tmp_path / "teste.csv"
    reading = Reading("24.1", 24.1, .9)
    with JsonlWriter(source) as log:
        log.write(make_record(1_800_000_000.5, "cliente São João", profile, reading, "ok"))
        log.write(make_record(1_800_000_001.5, "cliente São João", profile, reading, "unstable"))
    assert to_csv(source, target, ";") == 2
    with target.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert rows[0]["id-teste"] == "cliente São João"
    assert rows[0]["value"] == "24.1"
    assert rows[1]["value"] == ""
    assert rows[1]["status"] == "unstable"


def test_converter_accepts_the_four_original_fields_as_strings(tmp_path):
    source, target = tmp_path / "origem.txt", tmp_path / "saida.csv"
    source.write_text(json.dumps({"timestamp": "1800000000", "id-teste": "T1",
                                  "unidade medida": "volts", "value": "24.1"}) + "\n", encoding="utf-8")
    assert to_csv(source, target) == 1


@pytest.mark.parametrize("bad_line", [
    "{parcial", "[]", "null",
    '{"timestamp":1,"id-teste":"a","unidade medida":"volts","value":NaN}',
    '{"timestamp":1,"id-teste":"a","unidade medida":"volts","value":true}',
    '{"timestamp":1,"id-teste":"a","unidade medida":"volts","value":2,"status":"invalid"}',
    '{"timestamp":1,"id-teste":"a","unidade medida":"volts","value":null}',
])
def test_bad_lines_leave_existing_csv_untouched(tmp_path, bad_line):
    source, target = tmp_path / "origem.txt", tmp_path / "saida.csv"
    source.write_text("\n" + bad_line + "\n", encoding="utf-8")
    target.write_text("resultado anterior", encoding="utf-8")
    with pytest.raises(ValueError, match="Linha 2"):
        to_csv(source, target)
    assert target.read_text(encoding="utf-8") == "resultado anterior"
    assert list(tmp_path.glob(".display-csv-*")) == []


def test_converter_cannot_overwrite_its_input(tmp_path):
    source = tmp_path / "origem.txt"
    source.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError):
        to_csv(source, source)


def test_log_refuses_to_append_after_incomplete_line(tmp_path):
    source = tmp_path / "origem.txt"
    source.write_text('{"timestamp":', encoding="utf-8")
    with pytest.raises(ValueError):
        with JsonlWriter(source):
            pass
    assert source.read_text() == '{"timestamp":'
