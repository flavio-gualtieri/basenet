"""Filesystem layout + a tiny stage-timing helper shared by every CLI stage.

Every stage in this package prints rows-in / rows-out / wall-time (see
``StageTimer`` below) so a run can be sanity-checked from stdout alone.

All path constants below are computed live from `$BASENET_DATA_ROOT` on every
access (via module `__getattr__`, PEP 562) rather than frozen once at import
time. This is what lets tests point the whole pipeline at a temp directory
with `monkeypatch.setenv("BASENET_DATA_ROOT", tmp_path)` -- other modules
must do `from . import paths; paths.DATA_ROOT` (live attribute lookup) rather
than `from .paths import DATA_ROOT` (which would freeze the value at that
module's import time, before any test fixture gets a chance to run).
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional


def _repo_root() -> Path:
    # src/basenet/data/paths.py -> parents[3] is the repo root.
    return Path(__file__).resolve().parents[3]


def data_root() -> Path:
    return Path(os.environ.get("BASENET_DATA_ROOT", _repo_root() / "data"))


_DERIVED = {
    "DATA_ROOT": lambda r: r,
    "RAW_STATCAST_DIR": lambda r: r / "raw" / "statcast",
    "RAW_SCHEDULE_DIR": lambda r: r / "raw" / "schedule",
    "INTERIM_PA_DIR": lambda r: r / "interim" / "pa",
    "INTERIM_PLAYER_DAILY_DIR": lambda r: r / "interim" / "player_daily",
    "INTERIM_PITCHER_POST_GAME_DIR": lambda r: r / "interim" / "pitcher_post_game",
    "PROCESSED_MATCHUPS_DIR": lambda r: r / "processed" / "matchups",
    "PROCESSED_SPLIT_DIR": lambda r: r / "processed" / "split",
    "REFERENCE_DIR": lambda r: r / "reference",
    "PLAYER_IDS_PATH": lambda r: r / "reference" / "player_ids.parquet",
}


def __getattr__(name: str) -> Path:  # PEP 562: module-level lazy attributes
    if name in _DERIVED:
        return _DERIVED[name](data_root())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def raw_statcast_season_dir(season: int) -> Path:
    return __getattr__("RAW_STATCAST_DIR") / f"season={season}"


def raw_schedule_season_dir(season: int) -> Path:
    return __getattr__("RAW_SCHEDULE_DIR") / f"season={season}"


def interim_pa_season_dir(season: int) -> Path:
    return __getattr__("INTERIM_PA_DIR") / f"season={season}"


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class StageTimer:
    """Prints `<stage>: N in / M out (Ts)` on exit. See usage in each stage's __main__."""

    stage: str
    rows_in: Optional[int] = None
    rows_out: Optional[int] = None
    _t0: float = field(default_factory=time.perf_counter, init=False)

    @contextmanager
    def run(self) -> Iterator["StageTimer"]:
        self._t0 = time.perf_counter()
        try:
            yield self
        finally:
            dt = time.perf_counter() - self._t0
            rin = "?" if self.rows_in is None else f"{self.rows_in:,}"
            rout = "?" if self.rows_out is None else f"{self.rows_out:,}"
            print(f"[{self.stage}] rows_in={rin} rows_out={rout} wall_time={dt:.1f}s")
