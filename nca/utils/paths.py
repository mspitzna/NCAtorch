"""Helpers for creating run output directories."""

from datetime import datetime
from pathlib import Path


def create_run_folder(
    train_name: str,
    root: str | Path = "train_log",
    timestamp: datetime | None = None,
) -> str:
    """Atomically create a unique run folder ``<root>/<train_name>_<timestamp>``.

    Args:
        train_name: Base name of the run (``LOGGING.TRAIN_NAME``).
        root: Directory the run folder is created in.
        timestamp: Timestamp used for the name; defaults to now.

    Returns:
        Path of the newly created folder, as a string.
    """
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    base = Path(root) / f"{train_name}_{stamp}"
    candidate = base
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            candidate = base.with_name(f"{base.name}_{suffix}")
            suffix += 1
            continue
        return str(candidate)
