"""Tests for unique run-folder creation."""

import os
from datetime import datetime

from nca.utils.paths import create_run_folder

FIXED_TS = datetime(2024, 1, 2, 3, 4, 5)


def test_create_run_folder_uses_name_and_timestamp(tmp_path):
    folder = create_run_folder("run", root=str(tmp_path), timestamp=FIXED_TS)

    assert folder == str(tmp_path / "run_20240102_030405")
    assert (tmp_path / "run_20240102_030405").is_dir()


def test_create_run_folder_never_reuses_existing_folder(tmp_path):
    """Two runs with the same name and second must not share a folder."""
    folders = [
        create_run_folder("run", root=str(tmp_path), timestamp=FIXED_TS)
        for _ in range(3)
    ]

    assert len(set(folders)) == 3
    assert [folders[0], folders[1], folders[2]] == [
        str(tmp_path / "run_20240102_030405"),
        str(tmp_path / "run_20240102_030405_1"),
        str(tmp_path / "run_20240102_030405_2"),
    ]
    for folder in folders:
        assert os.path.isdir(folder)


def test_create_run_folder_keeps_existing_contents(tmp_path):
    first = create_run_folder("run", root=str(tmp_path), timestamp=FIXED_TS)
    marker = tmp_path / "run_20240102_030405" / "config.yaml"
    marker.write_text("keep me")

    second = create_run_folder("run", root=str(tmp_path), timestamp=FIXED_TS)

    assert first != second
    assert marker.read_text() == "keep me"
