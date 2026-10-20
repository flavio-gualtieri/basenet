"""matchups -> model-ready tensors.

    from basenet.data.datasets import (
        load_matchups, build_tensors, chronological_split, fit_scaler, Scaler,
    )

Given a matchups frame (or a game_pk-filtered slice of one), `build_tensors`
yields:

    lineup:          (N, 9, F_h)  ordered by slot, includes the (empirically
                                  measured, leakage-safe -- see matchups.py's
                                  compute_slot_pa_table) expected-PA-per-slot
                                  channel and a numeric handedness channel
    pa_weights:      (N, 9)       the SAME expected-PA values as their own
                                  raw (never scaled) tensor -- HitterEncoder
                                  needs actual PA counts for pooling weights,
                                  not z-scored ones; pass this as its
                                  `pa_weights` argument when pooling="pa_weighted"
    pitchers:        (N, K, F_p)  K = 1 + config.MAX_BULLPEN_SIZE. Index 0 is
                                  always the starter; 1..K-1 are available
                                  relievers (padded with mask=False, never
                                  NaN, when fewer are available)
    pitcher_mask:    (N, K)       True where a real pitcher, False where padding
    pitcher_weights: (N, K)       expected innings for that slot (SP:
                                  config.SP_EXPECTED_IP; bullpen: each's own
                                  IP share; 0 for padding)
    context:         (N, F_c)     batting_is_home, month, and cold-start flags
                                  (sp, lineup-mean, bullpen-mean-over-real-slots)
    park:            (N,)         int64 park index into the train-fit vocab
                                  (0 = unseen in train) -- categorical, for an
                                  embedding; never scaled
    y:               (N,)         runs_scored
    game_pk:         (N,)         carried through for group-aware splitting

No random splits anywhere: `chronological_split` assigns whole seasons to
train/val/test, which trivially satisfies "group by game_pk" (a season's
games never span a split boundary) and "chronological" (train's max date <
val's min date < test's min date), as required by the leakage tests.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from . import paths
from .features import (
    HITTER_CONTACT_COLUMNS, HITTER_FEATURE_COLUMNS, PITCHER_CONTACT_COLUMNS, PITCHER_FEATURE_COLUMNS,
    SLOT_PLATOON_COLUMNS,
)
from .matchups import SCHEDULE_CONTEXT_COLUMNS, WEATHER_CONTEXT_COLUMNS
from .paths import ensure_dir

HAND_MAP = {"L": -1.0, "R": 1.0, "S": 0.0}

CONTEXT_FEATURES = [
    "batting_is_home", "month",
    "sp_is_cold_start", "lineup_cold_start_frac", "bullpen_cold_start_frac",
]

K_PITCHERS = 1 + config.MAX_BULLPEN_SIZE  # starter + bullpen slots

# Optional feature groups, switched on in configs.yaml (`features.groups`).
# The pipeline always computes every column; these only choose which reach
# the tensors, so a group can be evaluated (or dropped) without a rebuild.
FEATURE_GROUPS: dict[str, str] = {
    "contact": "quality of contact: xwOBA / hard-hit / barrel rates (hitters produced, pitchers allowed)",
    "platoon": "per lineup slot: hitter's split vs the starter's hand + starter's split vs the hitter's side",
    "workload": "starter's recent batters faced per start: a context feature + sets SP/bullpen innings weights",
    "schedule": "day game, each team's days since last game, venue change, day-after-night",
    "weather": "game-time temperature, wind blowing out / in, indoors, precipitation",
}


def validate_groups(groups) -> tuple[str, ...]:
    groups = tuple(sorted(set(groups)))
    unknown = set(groups) - set(FEATURE_GROUPS)
    if unknown:
        raise ValueError(f"unknown feature group(s) {sorted(unknown)}; known: {sorted(FEATURE_GROUPS)}")
    return groups


def hitter_columns(groups=()) -> list[str]:
    """Per-slot hitter channels, before the expected_pa and handedness channels."""
    groups = validate_groups(groups)
    cols = [c for c in HITTER_FEATURE_COLUMNS if "contact" in groups or c not in HITTER_CONTACT_COLUMNS]
    return cols + (SLOT_PLATOON_COLUMNS if "platoon" in groups else [])


def pitcher_columns(groups=()) -> list[str]:
    """Per-pitcher channels (starter and bullpen alike), before handedness."""
    groups = validate_groups(groups)
    return [c for c in PITCHER_FEATURE_COLUMNS if "contact" in groups or c not in PITCHER_CONTACT_COLUMNS]


def context_columns(groups=()) -> list[str]:
    groups = validate_groups(groups)
    return (CONTEXT_FEATURES
            + (["sp_bf_per_start"] if "workload" in groups else [])
            + (SCHEDULE_CONTEXT_COLUMNS if "schedule" in groups else [])
            + (WEATHER_CONTEXT_COLUMNS if "weather" in groups else []))


def feature_dims(groups=()) -> tuple[int, int, int]:
    """(F_h, F_p, F_c) for these groups -- the model's input widths."""
    return (len(hitter_columns(groups)) + 2,   # + expected_pa + handedness
            len(pitcher_columns(groups)) + 1,  # + handedness
            len(context_columns(groups)))


# Widths with no optional groups (the base feature set).
F_HITTER, F_PITCHER, F_CONTEXT = feature_dims(())

# Tensors the Scaler actually normalizes. Everything else (pa_weights,
# pitcher_mask, pitcher_weights, y, game_pk) passes through untouched --
# pa_weights in particular must stay in raw, comparable-across-slots units
# for HitterEncoder's pooling to make sense as a weighted average.
SCALED_KEYS = ("lineup", "pitchers", "context")

# matchups column the park embedding is keyed on: StatsAPI's real venue id
# (not `park_id`, the home-team code -- which can't tell the Rays at
# Steinbrenner Field from the Rays at the Trop, or a London Series game
# from a home game).
PARK_KEY = "venue_id"
UNKNOWN_PARK = 0  # index for a park the train split never saw (or barely saw)
# A venue needs this many train rows (2 per game) to get its own embedding;
# one-off venues (Field of Dreams, Williamsport, international series) would
# otherwise get a vector fit to a handful of games. They share UNKNOWN_PARK.
PARK_MIN_TRAIN_ROWS = 40


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_matchups(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        p = paths.PROCESSED_MATCHUPS_DIR / f"matchups_{s}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(
            f"No matchups parquet found for seasons {seasons}. Run `python -m basenet.data.matchups` first."
        )
    return pd.concat(frames, ignore_index=True)


def build_park_vocab(df: pd.DataFrame, min_rows: int = PARK_MIN_TRAIN_ROWS) -> dict[str, int]:
    """Categorical index for `PARK_KEY`, starting at 1 (0 = UNKNOWN_PARK).
    Fit on the TRAIN split only (see fit_and_persist_split): a park that
    first appears in val/test would otherwise get an embedding row that no
    training example ever updated -- random noise at prediction time. It
    maps to UNKNOWN_PARK instead, as do parks with fewer than `min_rows`
    train rows. Keys are strings so the vocab survives a JSON round trip
    whatever the key column's dtype.
    """
    counts = df[PARK_KEY].dropna().astype(str).value_counts()
    keys = sorted(counts[counts >= min_rows].index.tolist())
    return {k: i for i, k in enumerate(keys, start=UNKNOWN_PARK + 1)}


def park_index(df: pd.DataFrame, park_vocab: dict[str, int]) -> np.ndarray:
    return df[PARK_KEY].astype(str).map(park_vocab).fillna(UNKNOWN_PARK).to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# Tensor construction
# ---------------------------------------------------------------------------

def _encode_hand(series: pd.Series) -> np.ndarray:
    return series.map(HAND_MAP).fillna(0.0).to_numpy(dtype=float)


def _hitter_slot_matrix(df: pd.DataFrame, slot: int, groups) -> np.ndarray:
    cols = [f"slot{slot}_{c}" for c in hitter_columns(groups)] + [f"slot{slot}_expected_pa"]
    arr = df[cols].to_numpy(dtype=float)
    hand = _encode_hand(df[f"slot{slot}_stands"])
    return np.concatenate([arr, hand[:, None]], axis=1)


def _pa_weights_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = [f"slot{s}_expected_pa" for s in range(1, config.N_LINEUP_SLOTS + 1)]
    return df[cols].to_numpy(dtype=float)


def _sp_vector(df: pd.DataFrame, groups) -> np.ndarray:
    cols = [f"sp_{c}" for c in pitcher_columns(groups)]
    arr = df[cols].to_numpy(dtype=float)
    hand = _encode_hand(df["sp_throws"])
    return np.concatenate([arr, hand[:, None]], axis=1)


def _bullpen_slot_vector(df: pd.DataFrame, k: int, groups) -> np.ndarray:
    cols = [f"bullpen{k}_{c}" for c in pitcher_columns(groups)]
    arr = df[cols].to_numpy(dtype=float)
    hand = _encode_hand(df[f"bullpen{k}_throws"])
    return np.concatenate([arr, hand[:, None]], axis=1)


def _starter_expected_ip(df: pd.DataFrame) -> np.ndarray:
    """The starter's recent batters faced per start, in innings, kept inside
    (1, TEAM_INNINGS_PER_GAME - 1) so neither side of the split vanishes."""
    ip = df["sp_bf_per_start"].to_numpy(dtype=float) / config.BATTERS_PER_INNING
    return np.clip(ip, 1.0, config.TEAM_INNINGS_PER_GAME - 1.0)


def build_pitcher_set(df: pd.DataFrame, groups=()) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stacks the starter (always index 0) with config.MAX_BULLPEN_SIZE
    bullpen slots into (N, K, F_p), plus (N, K) mask and (N, K) expected-
    innings weights -- the set PitcherEncoder consumes, replacing what used
    to be a single pre-aggregated composite. See matchups.py's
    resolve_bullpen_set for where the per-slot columns come from.

    Weights: without the "workload" group, a flat config.SP_EXPECTED_IP for
    every starter and config.BULLPEN_EXPECTED_TOTAL_IP shared by the
    bullpen. With it, the starter gets his own expected innings (from his
    recent batters faced per start) and the bullpen the rest of the game.
    """
    groups = validate_groups(groups)
    n = len(df)
    if "workload" in groups:
        sp_ip = _starter_expected_ip(df)
        bullpen_scale = (config.TEAM_INNINGS_PER_GAME - sp_ip) / config.BULLPEN_EXPECTED_TOTAL_IP
    else:
        sp_ip = np.full(n, config.SP_EXPECTED_IP, dtype=float)
        bullpen_scale = np.ones(n, dtype=float)
    vectors = [_sp_vector(df, groups)]
    masks = [np.ones(n, dtype=bool)]
    weights = [sp_ip]
    for k in range(1, config.MAX_BULLPEN_SIZE + 1):
        vectors.append(_bullpen_slot_vector(df, k, groups))
        masks.append(df[f"bullpen{k}_mask"].to_numpy(dtype=bool))
        # ip_share sums to BULLPEN_EXPECTED_TOTAL_IP over the real slots
        weights.append(df[f"bullpen{k}_ip_share"].to_numpy(dtype=float) * bullpen_scale)

    pitchers = np.stack(vectors, axis=1)   # (N, K, F_p)
    mask = np.stack(masks, axis=1)         # (N, K)
    weight = np.stack(weights, axis=1)     # (N, K)
    assert mask[:, 0].all(), (
        "index 0 of the pitcher set must always be the real starter -- "
        "PitcherEncoder's keep_sp_separate reads that slot unconditionally"
    )
    return pitchers, mask, weight


def _bullpen_cold_start_frac(df: pd.DataFrame) -> np.ndarray:
    """Mean is_cold_start over only the REAL (mask=True) bullpen slots for
    each row; 1.0 (fully cold) when no bullpen candidate was available at
    all, matching how an individual player's own debut is flagged.
    """
    cold = df[[f"bullpen{k}_is_cold_start" for k in range(1, config.MAX_BULLPEN_SIZE + 1)]].to_numpy(dtype=float)
    mask = df[[f"bullpen{k}_mask" for k in range(1, config.MAX_BULLPEN_SIZE + 1)]].to_numpy(dtype=bool)
    n_real = mask.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (cold * mask).sum(axis=1) / np.clip(n_real, 1, None)
    return np.where(n_real > 0, frac, 1.0)


def _context_matrix(df: pd.DataFrame, groups=()) -> np.ndarray:
    """Columns in context_columns(groups) order: the five base features
    (derived here), then any optional-group columns, read as-is."""
    is_home = df["batting_is_home"].astype(float).to_numpy()
    month = pd.to_datetime(df["game_date"]).dt.month.to_numpy(dtype=float)
    sp_cold = df["sp_is_cold_start"].astype(float).to_numpy()
    lineup_cold = df[[f"slot{s}_is_cold_start" for s in range(1, config.N_LINEUP_SLOTS + 1)]] \
        .astype(float).mean(axis=1).to_numpy()
    bullpen_cold = _bullpen_cold_start_frac(df)
    base = np.stack([is_home, month, sp_cold, lineup_cold, bullpen_cold], axis=1)
    extra = context_columns(groups)[len(CONTEXT_FEATURES):]
    if not extra:
        return base
    return np.concatenate([base, df[extra].to_numpy(dtype=float)], axis=1)


def build_tensors(df: pd.DataFrame, park_vocab: dict[str, int] | None = None,
                  groups=()) -> dict[str, np.ndarray]:
    """See the module docstring; `groups` (FEATURE_GROUPS) picks the optional
    feature columns -- the same groups must be used for every split and for
    the model's input widths (feature_dims)."""
    groups = validate_groups(groups)
    if park_vocab is None:
        park_vocab = build_park_vocab(df)
    lineup = np.stack(
        [_hitter_slot_matrix(df, s, groups) for s in range(1, config.N_LINEUP_SLOTS + 1)], axis=1
    )
    pitchers, pitcher_mask, pitcher_weights = build_pitcher_set(df, groups)
    return {
        "lineup": lineup,
        "pa_weights": _pa_weights_matrix(df),
        "pitchers": pitchers,
        "pitcher_mask": pitcher_mask,
        "pitcher_weights": pitcher_weights,
        "context": _context_matrix(df, groups),
        "park": park_index(df, park_vocab),
        "y": df["runs_scored"].to_numpy(dtype=float),
        "game_pk": df["game_pk"].to_numpy(),
    }


# ---------------------------------------------------------------------------
# Chronological, game_pk-grouped split (no random splits anywhere)
# ---------------------------------------------------------------------------

def chronological_split(
    seasons_train: list[int], season_val: int, season_test: int
) -> tuple[set[int], set[int], set[int]]:
    """Whole seasons -> whole splits. This trivially satisfies both leakage
    invariants: a game_pk belongs to exactly one season, so it can never
    appear in two splits, and each season's date range is disjoint from and
    earlier/later than its neighbors, so max(train date) < min(val date) <
    min(test date) holds by construction (asserted below as a safety check,
    not as a hopeful assumption).
    """
    train_df = load_matchups(seasons_train)
    val_df = load_matchups([season_val])
    test_df = load_matchups([season_test])

    if not (train_df["game_date"].max() < val_df["game_date"].min()):
        raise ValueError("train/val date ranges overlap -- seasons_train and season_val must not overlap")
    if not (val_df["game_date"].max() < test_df["game_date"].min()):
        raise ValueError("val/test date ranges overlap -- season_val and season_test must not overlap")

    return (set(train_df["game_pk"]), set(val_df["game_pk"]), set(test_df["game_pk"]))


# ---------------------------------------------------------------------------
# Scaler (StandardScaler-style: fit on train only, persisted alongside the split)
# ---------------------------------------------------------------------------

@dataclass
class Scaler:
    mean: dict[str, np.ndarray] = field(default_factory=dict)
    std: dict[str, np.ndarray] = field(default_factory=dict)
    park_vocab: dict[str, int] = field(default_factory=dict)

    def transform(self, tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = dict(tensors)
        for key in SCALED_KEYS:
            out[key] = (tensors[key] - self.mean[key]) / self.std[key]
        return out

    def save(self, path: Path) -> None:
        ensure_dir(path.parent)
        payload = {f"mean__{k}": v for k, v in self.mean.items()}
        payload.update({f"std__{k}": v for k, v in self.std.items()})
        np.savez(path.with_suffix(".npz"), **payload)
        path.with_suffix(".park_vocab.json").write_text(json.dumps(self.park_vocab))

    @classmethod
    def load(cls, path: Path) -> "Scaler":
        data = np.load(path.with_suffix(".npz"))
        mean, std = {}, {}
        for key in data.files:
            kind, name = key.split("__", 1)
            (mean if kind == "mean" else std)[name] = data[key]
        park_vocab = json.loads(path.with_suffix(".park_vocab.json").read_text())
        return cls(mean=mean, std=std, park_vocab=park_vocab)


def fit_scaler(train_tensors: dict[str, np.ndarray], park_vocab: dict[str, int]) -> Scaler:
    mean, std = {}, {}
    for key in SCALED_KEYS:
        arr = train_tensors[key]
        if key == "pitchers":
            # mask-aware: padding rows carry a finite placeholder (0.0), not
            # a sample of real pitcher data -- including them would skew the
            # mean/std toward 0 for every stat. Fit on real (mask=True) rows
            # only; transform still applies uniformly (see docstring above).
            mask = train_tensors["pitcher_mask"]
            flat = arr.reshape(-1, arr.shape[-1])
            real = flat[mask.reshape(-1)]
            m = np.nanmean(real, axis=0)
            s = np.nanstd(real, axis=0)
        else:
            axis = tuple(range(arr.ndim - 1))  # pool over game (and slot, for lineup) axes
            m = np.nanmean(arr, axis=axis)
            s = np.nanstd(arr, axis=axis)
        s = np.where(s < 1e-8, 1.0, s)
        mean[key], std[key] = m, s
    return Scaler(mean=mean, std=std, park_vocab=park_vocab)


def fit_and_persist_split(
    seasons_train: list[int], season_val: int, season_test: int,
    out_dir: Path | None = None, feature_groups=(),
) -> Scaler:
    """Computes the chronological split, fits the scaler on the train
    game_pks ONLY, and persists both the split (game_pk lists) and the
    scaler to `out_dir` (default: paths.PROCESSED_SPLIT_DIR, resolved live so
    tests pointed at a scratch BASENET_DATA_ROOT land there too) so a later
    run reuses the exact same normalization.
    """
    if out_dir is None:
        out_dir = paths.PROCESSED_SPLIT_DIR
    train_pks, val_pks, test_pks = chronological_split(seasons_train, season_val, season_test)
    all_df = load_matchups(seasons_train + [season_val, season_test])
    train_df = all_df[all_df["game_pk"].isin(train_pks)]
    park_vocab = build_park_vocab(train_df)  # train-only, like the scaler
    train_tensors = build_tensors(train_df, park_vocab, feature_groups)
    scaler = fit_scaler(train_tensors, park_vocab)

    ensure_dir(out_dir)
    scaler.save(out_dir / "scaler")
    (out_dir / "split.json").write_text(json.dumps({
        "train": sorted(train_pks), "val": sorted(val_pks), "test": sorted(test_pks),
        "seasons_train": seasons_train, "season_val": season_val, "season_test": season_test,
        "feature_groups": list(validate_groups(feature_groups)),
    }))
    return scaler
