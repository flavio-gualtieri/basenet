"""Strict YAML -> dataclass config loading for the model layer.

`configs.yaml` is the single origin point for every model hyperparameter.
Each of `model.encoder.hitter`, `model.encoder.pitcher`, and `model.head`
must supply EXACTLY the fields its dataclass expects -- no more, no less --
with one deliberate exception: `in_dim` (for both encoders) is never a yaml
field. It's derived from the data layer's own feature counts
(`basenet.data.datasets.F_HITTER` / `F_PITCHER`) and injected here, so it
can never silently drift out of sync with `features.py`'s actual column
list. A yaml section missing a required field, carrying a misspelled one,
or carrying `in_dim` by hand, raises -- it does not fall back to the
dataclass's own Python-side default, and it does not let a stale/guessed
`in_dim` silently produce a shape mismatch three layers down.

Named distinctly from `basenet.data.config` (data-pipeline constants) --
this is model-architecture config, a different thing entirely.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from .data import datasets
from .encoder.hitter import HitterEncoderConfig
from .encoder.pitcher import PitcherEncoderConfig
from .head.head import HeadConfig

# src/basenet/model_config.py -> parents[2] is the repo root.
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs.yaml"


@dataclass(frozen=True)
class ModelConfig:
    hitter: HitterEncoderConfig
    pitcher: PitcherEncoderConfig
    head: HeadConfig
    context_dim: int  # F_CONTEXT -- carried alongside so BaseNet needs no import from basenet.data
    n_parks: int      # size of the train-fit park vocab (excluding the UNKNOWN_PARK index 0)


def _validate_and_build(cfg_cls: type, section_name: str, raw: dict[str, Any],
                         injected: dict[str, Any]) -> Any:
    """One yaml section -> one dataclass instance, strictly: `raw`'s keys
    must exactly match `cfg_cls`'s fields minus whatever `injected` already
    supplies. `raw` carrying an injected key too (e.g. a hand-typed
    `in_dim`) is flagged as unexpected, not silently overridden -- two
    competing sources of truth for the same value is exactly the failure
    mode this is meant to catch.
    """
    expected = {f.name for f in fields(cfg_cls)} - set(injected)
    got = set(raw)

    missing = expected - got
    extra = got - expected
    if missing or extra:
        problems = []
        if missing:
            problems.append(f"missing: {sorted(missing)}")
        if extra:
            problems.append(f"unexpected: {sorted(extra)}")
        raise ValueError(
            f"configs.yaml model.{section_name} does not match {cfg_cls.__name__}'s "
            f"fields ({'; '.join(problems)}). Every field must be given explicitly, "
            f"except {sorted(injected) or '(none)'}, which come from the data layer, "
            f"never from yaml."
        )
    return cfg_cls(**raw, **injected)


def load_model_config(n_parks: int, path: Path | str = DEFAULT_CONFIG_PATH) -> ModelConfig:
    """`n_parks` is the train split's park vocab size (`len(scaler.park_vocab)`):
    data-derived per split, so it can't be a yaml value either."""
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)

    try:
        model = raw["model"]
        hitter_raw = model["encoder"]["hitter"]
        pitcher_raw = model["encoder"]["pitcher"]
        head_raw = model["head"]
    except KeyError as e:
        raise ValueError(f"{path} is missing expected section {e}") from e

    hitter = _validate_and_build(
        HitterEncoderConfig, "encoder.hitter", hitter_raw, {"in_dim": datasets.F_HITTER},
    )
    pitcher = _validate_and_build(
        PitcherEncoderConfig, "encoder.pitcher", pitcher_raw, {"in_dim": datasets.F_PITCHER},
    )
    head = _validate_and_build(HeadConfig, "head", head_raw, {})

    return ModelConfig(hitter=hitter, pitcher=pitcher, head=head, context_dim=datasets.F_CONTEXT,
                       n_parks=n_parks)
