import os

from piper_handeye.retention import prune_capture_groups, prune_old_entries


def test_prune_old_entries_keeps_newest_twenty(tmp_path):
    for index in range(25):
        entry = tmp_path / f"run_{index:02d}"
        entry.mkdir()
        os.utime(entry, (index, index))
    prune_old_entries(tmp_path, keep=20)
    assert len(list(tmp_path.iterdir())) == 20
    assert not (tmp_path / "run_00").exists()
    assert (tmp_path / "run_24").exists()


def test_capture_rgb_and_depth_files_count_as_one_action(tmp_path):
    for index in range(23):
        stem = f"20260902_170000_{index:06d}"
        for suffix in (".jpg", "_depth_mm.png", "_depth_color.png"):
            entry = tmp_path / f"{stem}{suffix}"
            entry.touch()
            os.utime(entry, (index, index))
    prune_capture_groups(tmp_path, keep=20)
    assert len(list(tmp_path.iterdir())) == 60
    assert not (tmp_path / "20260902_170000_000000.jpg").exists()
    assert (tmp_path / "20260902_170000_000022.jpg").exists()
