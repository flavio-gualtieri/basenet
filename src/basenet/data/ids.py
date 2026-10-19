"""MLBAM-canonical player-ID crosswalk.

statcast pitch rows already key `batter`/`pitcher` by MLBAM ID, so nothing
downstream *needs* this crosswalk to join pitches to players. It exists so
that (a) any future join against a name-keyed or FanGraphs-keyed source has
a canonical, persisted mapping to go through instead of a fresh name match,
and (b) humans reading `data/reference/player_ids.parquet` can see who a
given MLBAM ID actually is. Never join raw data on `player_name` -- names
collide (there are multiple "Will Smith"s active in this window) and are
formatted inconsistently across sources.
"""
from __future__ import annotations

import pandas as pd

from . import paths
from .paths import ensure_parent

CROSSWALK_COLUMNS = [
    "key_mlbam", "key_bbref", "key_fangraphs", "key_retro",
    "name_first", "name_last", "mlb_played_first", "mlb_played_last",
]


def build_id_crosswalk(mlbam_ids: set[int]) -> pd.DataFrame:
    """Reverse-lookup a set of MLBAM ids to the Chadwick Bureau register.

    Network call (Chadwick register CSV, cached by `pybaseball.cache`).
    """
    import pybaseball as pb  # imported lazily: only ingest.py needs network

    if not mlbam_ids:
        return pd.DataFrame(columns=CROSSWALK_COLUMNS)

    df = pb.playerid_reverse_lookup(sorted(mlbam_ids), key_type="mlbam")
    missing = mlbam_ids - set(df["key_mlbam"].dropna().astype(int))
    if missing:
        print(f"[ids] {len(missing)} MLBAM id(s) not found in Chadwick register "
              f"(kept out of crosswalk, still usable via raw id): {sorted(missing)[:10]}...")
    return df[[c for c in CROSSWALK_COLUMNS if c in df.columns]]


def load_crosswalk() -> pd.DataFrame:
    if not paths.PLAYER_IDS_PATH.exists():
        return pd.DataFrame(columns=CROSSWALK_COLUMNS)
    return pd.read_parquet(paths.PLAYER_IDS_PATH)


def upsert_crosswalk(new_rows: pd.DataFrame) -> pd.DataFrame:
    """Merge newly-resolved ids into the persisted crosswalk, keyed on key_mlbam."""
    existing = load_crosswalk()
    # skip the concat entirely on the first-ever ingest (existing is empty):
    # pandas emits a FutureWarning about dtype inference across an all-empty
    # frame otherwise, and there's nothing to merge in that case anyway.
    combined = new_rows if existing.empty else pd.concat([existing, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["key_mlbam"], keep="last")
    combined = combined.sort_values("key_mlbam").reset_index(drop=True)
    ensure_parent(paths.PLAYER_IDS_PATH)
    combined.to_parquet(paths.PLAYER_IDS_PATH, index=False)
    return combined
