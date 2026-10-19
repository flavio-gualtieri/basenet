"""Run every leakage + reconciliation check against real, already-produced
pipeline output for the given seasons. Exits non-zero on any failure.

    python -m basenet.data.validate --seasons 2021 2022 2023 2024 2025

This is deliberately the same logic as `tests/data/test_leakage.py`'s named
checks (test_runs_reconcile, test_two_rows_per_game), run here against the
real `data/` tree instead of the synthetic pytest fixture, plus a check of
every training label against StatsAPI's official final scores, plus the full
pytest suite (which additionally covers the split-mechanism checks --
test_game_grouping, test_chronological, test_scaler_fit_scope -- that don't
depend on which seasons are real vs synthetic).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from . import config
from .pa import load_played_schedule, reconcile_runs
from .paths import interim_pa_season_dir, PROCESSED_MATCHUPS_DIR, StageTimer


def validate_runs_reconcile(seasons: list[int]) -> list[str]:
    failures = []
    for season in seasons:
        p = interim_pa_season_dir(season) / f"pa_{season}.parquet"
        if not p.exists():
            failures.append(f"season {season}: missing {p} -- run `python -m basenet.data.pa` first")
            continue
        df = pd.read_parquet(p)
        bad = reconcile_runs(df)
        if not bad.empty:
            failures.append(f"season {season}: runs reconciliation failed for {len(bad)} team-game(s)")
    return failures


def validate_labels_against_schedule(seasons: list[int]) -> list[str]:
    """`matchups.runs_scored` (the training label) vs StatsAPI's official
    final score -- the only check here that uses a source independent of
    statcast's own score columns."""
    failures = []
    for season in seasons:
        p = PROCESSED_MATCHUPS_DIR / f"matchups_{season}.parquet"
        if not p.exists():
            continue  # reported by validate_two_rows_per_game
        try:
            sched = load_played_schedule(season)
        except FileNotFoundError as e:
            failures.append(f"season {season}: {e}")
            continue
        df = pd.read_parquet(p, columns=["game_pk", "batting_is_home", "runs_scored"])
        merged = df.merge(sched[["game_pk", "home_score", "away_score"]], on="game_pk", how="left")
        official = merged["home_score"].where(merged["batting_is_home"], merged["away_score"])
        bad = merged[official.isna() | (official != merged["runs_scored"])]
        if not bad.empty:
            failures.append(f"season {season}: {bad['game_pk'].nunique()} game(s) whose runs_scored "
                            f"label disagrees with (or is missing from) the StatsAPI schedule")
    return failures


def validate_two_rows_per_game(seasons: list[int]) -> list[str]:
    failures = []
    for season in seasons:
        p = PROCESSED_MATCHUPS_DIR / f"matchups_{season}.parquet"
        if not p.exists():
            failures.append(f"season {season}: missing {p} -- run `python -m basenet.data.matchups` first")
            continue
        df = pd.read_parquet(p)
        counts = df.groupby("game_pk").size()
        bad_count = counts[counts != 2]
        if not bad_count.empty:
            failures.append(f"season {season}: {len(bad_count)} game_pk(s) without exactly 2 rows")
            continue
        for game_pk, g in df.groupby("game_pk"):
            if g["batting_team"].nunique() != 2:
                failures.append(f"season {season}: game_pk {game_pk} rows don't have distinct batting_team")
    return failures


def run_pytest_suite() -> int:
    # resolve relative to the repo root, not the caller's cwd, so `validate`
    # works the same regardless of where it's invoked from.
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run([sys.executable, "-m", "pytest", "tests/data", "-q"], cwd=repo_root)
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True, choices=config.SEASONS)
    parser.add_argument("--skip-pytest", action="store_true",
                         help="skip the pytest suite (game_pk grouping / chronological / "
                              "scaler-fit-scope checks); real-data checks still run")
    args = parser.parse_args(argv)

    all_failures: list[str] = []
    with StageTimer(f"validate seasons={args.seasons}").run() as t:
        all_failures += validate_runs_reconcile(args.seasons)
        all_failures += validate_two_rows_per_game(args.seasons)
        all_failures += validate_labels_against_schedule(args.seasons)
        t.rows_in = len(args.seasons)
        t.rows_out = len(args.seasons) - len(all_failures)

    for f in all_failures:
        print(f"[validate] FAIL: {f}")

    pytest_rc = 0
    if not args.skip_pytest:
        pytest_rc = run_pytest_suite()

    if all_failures or pytest_rc != 0:
        print(f"[validate] {len(all_failures)} real-data failure(s), pytest_rc={pytest_rc}")
        return 1
    print("[validate] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
