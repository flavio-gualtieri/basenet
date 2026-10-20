# BaseNet

Predicting how many runs an MLB team scores in a game, from the players who
take the field. The project has two parts: a leakage-safe data pipeline that
builds a pre-game feature table from raw Statcast pitch data, and a neural
model (BaseNet) evaluated against a linear model trained on the same inputs.

**Headline result:** on a held-out 2025 season, BaseNet matches a linear
count GLM but doesn't beat it. Both clearly beat a constant prediction, and
most of the remaining error is game-to-game randomness that no pre-game model
can remove. The pipeline and the evaluation are the substance of the project;
the [results](#results) section explains the numbers.

## Contents

- [The problem](#the-problem)
- [Results](#results)
- [Data pipeline](#data-pipeline)
- [Model](#model)
- [Evaluation](#evaluation)
- [Reproducing](#reproducing)
- [Repository layout](#repository-layout)
- [Limitations and next steps](#limitations-and-next-steps)

## The problem

Each training example is one team's turn at bat in one game: its starting
lineup against the opposing starter and bullpen, in a given park and
weather. The label is the runs that team scored. Every game gives two rows,
one per side.

| | |
|---|---|
| Seasons | 2021–2025 regular season (2020's 60-game season is excluded) |
| Games / rows | 12,148 games → 24,296 team-game rows |
| Split | train 2021–23 (14,578 rows) · val 2024 (4,858) · test 2025 (4,860) |
| Label | runs scored: mean ≈ 4.5, variance ≈ 10 |

Variance is roughly twice the mean, so a Poisson model is the wrong shape.
The model and both baselines predict a full **negative binomial**
distribution over runs, not a single number.

## Results

Test season 2025. BaseNet is reported as mean ± sd over 5 training seeds
(`make seeds`). The baselines are deterministic.

| model | test NLL ↓ | P(actual runs) ↑ | MAE ↓ |
|---|---|---|---|
| Constant (training mean, NB) | 2.4730 | 8.43% | 2.547 |
| Linear GLM (NB, same inputs) | **2.4636** | **8.51%** | 2.526 |
| BaseNet (NB) | 2.4655 ± 0.0009 | 8.50% | **2.506 ± 0.004** |

- **NLL** is the negative log-likelihood of the actual run total, in nats per
  team-game. It's the primary metric because it scores the whole predicted
  distribution.
- **P(actual runs)** is `exp(-NLL)`: the geometric-mean probability the model
  assigned to the score that actually happened.
- **MAE** is the absolute error of the predicted mean.

How to read this:

- **Both learned models beat the constant**, by about 0.009 nats per game.
  That's a small but real signal: who plays, where and in what conditions
  shifts the expected score.
- **BaseNet doesn't beat the GLM on likelihood.** It trails by
  +0.0019 ± 0.0009 nats, and the GLM won in all 5 seeds. With about 14.6k
  training rows, the network's extra capacity doesn't find structure that the
  linear model misses.
- **BaseNet has a lower MAE** in every seed. Its mean predictions are slightly
  better, but its predicted distribution isn't. MAE only scores the mean,
  so it's the secondary metric.
- **The ceiling is low.** Even a perfect model of the inputs can't predict a
  single game's runs much better than this, because baseball scoring is
  noisy: a team's run total has a standard deviation of about 3.2 against a
  mean of about 4.5.

![Training curve: train/val NLL and val MAE per epoch, against the constant
and GLM baselines](docs/training_curve.png)

*One run (seed 42). Early stopping restores the best-val epoch, and test is
scored once on that model. Train loss sits above val partly because dropout
is on during training.*

### What was tried

- **Poisson → negative binomial:** the largest single improvement. Test
  P(actual) went from 6.8% to 8.5% for BaseNet. The baselines were switched
  to NB at the same time, so the comparison stays like for like.
- **Optional feature groups** (contact quality, platoon splits, starter
  workload, schedule/travel, weather): each group was ablated on the 2024
  validation season over 3 seeds. Only **platoon** and **weather** helped
  both BaseNet and the GLM, so they're on by default
  ([configs.yaml](configs.yaml)). The others were within seed noise and stay
  available behind the config switch.

## Data pipeline

`ingest → pa → features → matchups → validate`, under
[src/basenet/data/](src/basenet/data/). [DATA.md](DATA.md) documents every
column and the exact time window each feature is computed over.

1. **ingest:** snapshots raw Statcast pitch data (via `pybaseball`) and the
   MLB StatsAPI schedule (official scores, venue ids, game-time weather) to
   parquet. This is the only stage that touches the network. Every later
   stage is a pure function of these files.
2. **pa:** about 900k plate appearances. Labels come from the **official**
   StatsAPI final score, cross-checked against Statcast's own play-by-play
   (walk-offs on a wild pitch or balk are the edge case Statcast misses).
3. **features:** per-player, per-game rolling rates (strikeout, walk, HR,
   batted-ball mix, contact quality, platoon splits, fatigue). They're
   computed in DuckDB SQL windows that end **strictly before** the game, and
   shrunk toward a running league mean (empirical Bayes) so small samples
   don't produce extreme values.
4. **matchups:** one row per team-game. It holds the 9-man lineup in batting
   order, the starter, and the set of available relievers. Bullpen
   availability is inferred from recent appearances and pitch counts, with
   rotation starters excluded. Park is keyed on the real ballpark, not the
   home team, so neutral-site and temporary-home games resolve correctly.
5. **validate:** re-checks the real data tree. Run totals reconcile with
   official scores, every game has exactly two rows, and every label matches
   StatsAPI.

**Leakage is the main design constraint.** It's enforced by tests, not only
by convention:

- Every feature uses only games before the target game, including the second
  game of a doubleheader.
- League means are as of the end of the previous day, so April's shrinkage
  never sees September.
- Splits are by whole season; a game's two rows never land in different
  splits.
- The scaler and the park vocabulary are fit on train only.

[tests/data/test_leakage.py](tests/data/test_leakage.py) checks each of these
end to end on a small synthetic season. For example, it recomputes the
rolling counts for every player-game from scratch and asserts equality with
the stored values.

## Model

[src/basenet/model.py](src/basenet/model.py), about 74k parameters.

- **Hitter encoder:** a 1-D convolution over the 9 lineup slots with
  *circular* padding, because the order wraps from the 9th hitter back to
  the leadoff. It's then pooled across slots, so each hitter is seen next to
  the hitters around him.
- **Pitcher encoder:** a shared MLP applied to each pitcher (the starter plus
  up to 12 masked relievers), then pooled weighted by expected innings. The
  starter's vector is also kept separately. The pooling is
  permutation-invariant over the bullpen.
- **Head:** an MLP over [hitters, pitchers, context, learned park
  embedding]. It outputs the log-mean of a negative binomial with a learned
  dispersion. Unseen parks map to a fixed zero embedding.

Training uses AdamW with early stopping on val NLL. The whole table fits in
memory, so there is no `DataLoader`. A training run takes about 10 s on an
Apple-silicon laptop.

## Evaluation

- **Chronological split by season.** Validation (2024) picks the epoch, the
  distribution and the feature groups. Test (2025) is never used for
  selection.
- **Baselines use the same tensors and likelihood** as BaseNet
  ([baseline.py](src/basenet/baseline.py)). Comparing a negative-binomial
  network against a Poisson baseline would credit the network with the
  better distribution. The GLM uses the lineup-mean hitter features, the
  starter, the innings-weighted bullpen mean, context and park one-hots.
- **Multiple seeds** ([seeds.py](src/basenet/seeds.py)) put the
  BaseNet-vs-GLM gap next to seed-to-seed noise instead of quoting one lucky
  run.

## Reproducing

Requires Python ≥ 3.11.

```bash
make install                  # pip install -e ".[dev]"
make test                     # 64 tests, synthetic data, no network, ~30 s
make data                     # ingest -> pa -> features -> matchups -> validate
make train                    # one run -> runs/summary.json, runs/training_curve.png
make seeds                    # 5 seeds -> runs/seeds/seeds_summary.json (the table above)
```

- `make data` downloads about 450 MB of Statcast data. That download is the
  only slow step, and it's idempotent: an interrupted run resumes from the
  seasons already on disk. After that, the offline stages take about 2 min
  on a laptop (most of it in `matchups`), and `make seeds` takes about
  75 s.
- The offline stages are deterministic. Re-running them from the same raw
  snapshot reproduces identical tables.
- To use a specific interpreter: `make train PYTHON=/path/to/python`.
- To point the pipeline at another data directory: `BASENET_DATA_ROOT=...`.
- `data/` and `runs/` are git-ignored and regenerated by the steps above.

## Repository layout

```
src/basenet/
  data/            pipeline stages (ingest, pa, features, matchups, validate),
                   tensor building + scaling (datasets.py), constants (config.py)
  encoder/         hitter (circular conv) and pitcher (masked set) encoders
  head/            NB / Poisson output head
  model.py         BaseNet = encoders + park embedding + head
  baseline.py      constant + linear GLM baselines
  distributions.py Poisson / negative-binomial log-likelihoods
  train.py         one training run
  seeds.py         multi-seed runner + aggregate
tests/
  conftest.py      synthetic two-team season run through the real pipeline
  data/            pipeline, feature and leakage tests
  model/           distribution math (vs scipy) and end-to-end training smoke tests
DATA.md            column-by-column data contract with feature time windows
configs.yaml       model hyperparameters and feature groups
```

## Limitations and next steps

- **Lineups are the ones that actually played**, not the lineups announced a
  few hours before first pitch. Starting lineups are usually public by then,
  but the model also sees late scratches and starter changes that a real
  pre-game forecast wouldn't know about. Fixing this properly needs a
  historical source of announced lineups.
- **Bullpen availability is a heuristic.** It looks at recent appearances
  and pitch counts. It isn't the real roster, so it can miss a reliever just
  activated from the injured list.
- **Data volume.** Five seasons is small for a neural network. Natural next
  steps:
  - add 2015–2019;
  - add an auxiliary per-plate-appearance outcome loss to give the encoders
    far more supervision than one run total per game;
  - add plate-discipline, pitch-quality, defence and umpire features.
- **An external benchmark** worth adding: betting-market run totals.

## Data sources

[Baseball Savant / Statcast](https://baseballsavant.mlb.com/) via
[pybaseball](https://github.com/jldbc/pybaseball), and the public
[MLB StatsAPI](https://statsapi.mlb.com/). This repository doesn't
redistribute any data.

## License

[MIT](LICENSE)
