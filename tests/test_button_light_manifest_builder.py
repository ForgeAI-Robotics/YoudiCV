import csv
import sys

from button_light_evaluation import make_manifest


def test_builds_manifest_from_state_directories(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset"
    (dataset / "ON").mkdir(parents=True)
    (dataset / "OFF").mkdir()
    (dataset / "ON" / "a.jpg").write_bytes(b"a")
    (dataset / "OFF" / "b.png").write_bytes(b"b")
    (dataset / "OFF" / "ignore.txt").write_text("x")
    output = tmp_path / "manifest.csv"
    monkeypatch.setattr(sys, "argv", ["make_manifest", str(dataset), "--target", "10", "--output", str(output)])
    assert make_manifest.main() == 0
    with output.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [(row["target"], row["state"]) for row in rows] == [("10", "ON"), ("10", "OFF")]
