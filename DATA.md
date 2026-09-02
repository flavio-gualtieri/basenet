# DATA.md — the `matchups` training table

This documents every column produced by the data pipeline (`src/basenet/data/`),
with special attention to **the exact time window each feature is computed
over** — that's the part that determines whether the table is safe to train
on. See each module's docstring for the mechanics; this file is the contract.

Pipeline: `ingest -> pa -> features -> matchups -> datasets`. Run order and
CLI are in the root spec / each module's `--help`.

## Conventions used throughout

- **Ordering key**: `(game_date, game_pk, at_bat_number)`. Two games can
  share a `game_date` (doubleheader); `game_pk` breaks that tie. This is an
  approximation (statcast's timestamp field, `sv_id`, is entirely null in
  modern data) but is applied consistently everywhere a "which game came
  first" decision is made, so it's at least internally coherent.
- **"As of the day before this game"**: in practice, "as of strictly before
  this row's `(game_date, game_pk)` in the ordering key above" — for a
  doubleheader's second game, that includes the first game, which is
  correct (it already happened).
- **park_id is a team-code proxy, not a real venue id.** Standard statcast
  pulls carry no venue/ballpark id column. `park_id` = the home team's code.
  This misattributes park effects whenever a team's home games are actually
  played elsewhere (the Rays' 2025 season at Steinbrenner Field; London/
  Mexico City/Seoul/Tokyo series). Known v1 limitation — a v2 would join
  MLB StatsAPI's schedule for a real venue id per `game_pk`.
- **v1 uses actual lineups/starters**, not lineups as known a few hours
  pre-game. Fine for training; not representative of true pre-game
  inference availability.

---

## `pa` (`data/interim/pa/season=YYYY/*.parquet`)

One row per plate appearance (the terminal pitch of each `at_bat_number`).

| column | meaning | window |
|---|---|---|
| `game_pk`, `game_date`, `season` | game identity | n/a (known pre-game) |
| `park_id` | home team code (see caveat above) | n/a |
| `is_doubleheader_game_2` | True iff this is the later of two same-date game_pks for this team pair (game_pk-ordinal approximation) | n/a |
| `inning`, `half` | inning number, `Top`/`Bot` | this PA |
| `batting_team`, `pitching_team` | derived from `half` + park_id | this PA |
| `batter_id`, `pitcher_id` | MLBAM ids | this PA |
| `batter_stands`, `pitcher_throws` | L/R/S — static, known pre-game | this PA |
| `order_slot` | 1-9, via `((rank_of_this_team's_PAs_so_far - 1) % 9) + 1` | this PA (see pa.py docstring: this is the *exact* rule, not an approximation, given a complete/correctly-ordered PA sequence) |
| `times_through_order` | statcast's own `n_thruorder_pitcher`, taken as-is | this PA |
| `outcome` | K / BB / HBP / OUT_IP / 1B / 2B / 3B / HR / ROE / OTHER | this PA (see config.EVENTS_TO_OUTCOME for the exact statcast `events` mapping; fielder's-choice and catcher's-interference bucket into OTHER) |
| `runs_on_play` | `post_bat_score - bat_score` | this PA |
| `bb_type` | ground_ball/fly_ball/line_drive/popup, only for balls in play | this PA |
| `at_bat_number` | internal ordering aid, not a model feature | n/a |
| `final_home_score`, `final_away_score` | box score, carried on every row of the game for the runs-reconciliation check | end of game (**label-adjacent — never used as a pre-game feature**, only for validation) |

**Label leakage note:** `final_home_score`/`final_away_score` exist purely
so `pa.reconcile_runs` (and `validate.py`) can check "do the summed
`runs_on_play` equal the box score" without a second pass over raw pitches.
They are outcome columns for the *current* game and must never be joined
into a feature for the model — matchups.py only ever reads them to compute
the label (`runs_scored`), never a feature.

---

## `player_daily` (`data/interim/player_daily/*.parquet`)

Keyed on **(`player_id`, `role`, `game_date`, `game_pk`)** — `role` is added
beyond the spec's literal 3-part key because two-way players (a pitcher who
also hits, in this window) need independent hitter/pitcher rows rather than
colliding.

Every stat below is computed from PAs **strictly before** this row's
`(game_date, game_pk)` — see features.py's module docstring for the exact
windowing mechanism (PA-granularity cumulative SQL windows, collapsed to the
first PA of each player-game).

| column | meaning | window |
|---|---|---|
| `std_pa` / `std_bf` | count of PAs (hitter) / batters faced (pitcher) | season-to-date, strictly before this game |
| `trailing_pa` / `trailing_bf` | same count | trailing 300 PA/BF, strictly before this game, spans season boundaries |
| `std_{stat}`, `trailing_{stat}` for `stat` in: hitters — `k_pct, bb_pct, hbp_pct, x1b_rate, x2b_rate, x3b_rate, hr_rate, babip, gb_pct, fb_pct`; pitchers — `k_pct, bb_pct, hr_per_bf, gb_pct, contact_pct, whiff_pct` | empirical-Bayes-shrunk rate: `(observed_count + k*league_rate) / (n + k)`, `k` from `config.HITTER_SHRINKAGE_K` / `PITCHER_SHRINKAGE_K` | same STD / trailing window as the count above; `league_rate` is the running league mean for that stat/role/season through the **end of the prior calendar date** (see "league mean" note below) |
| `is_cold_start` | `std_pa == 0` (hitter) / `std_bf == 0` (pitcher) — literal debut in the ingested window | n/a (flag, not a rate) |
| `days_since_last_game` (hitter) / `days_rest` (pitcher) | calendar days since the player's previous game/outing; default (`config.DEFAULT_DAYS_SINCE_LAST_GAME` / `DEFAULT_DAYS_REST`) on a debut | up to and including the prior appearance |
| `pitch_count_last_outing` (pitcher) | total pitches thrown in the immediately preceding outing | prior outing only |
| `pitch_count_trailing_3d` (pitcher) | total pitches thrown in the trailing 3 calendar days | strictly before this game |

**League-mean note:** the shrinkage target is *not* a whole-season constant
(that would leak the season's future games into April's shrinkage) and
*not* recomputed per-PA (unnecessary precision for an aggregate). It's a
per-calendar-date running mean, computed the same leakage-safe way as the
player-level STD window but pooled across every player of that role,
consulted at the granularity of "through the end of yesterday." A small
static fallback (`config.LEAGUE_MEAN_FALLBACK_HITTER/PITCHER`) covers the
otherwise-undefined first date(s) of the very first ingested season, when
no real league sample exists yet.

**Handedness is not stored here.** `batter_stands`/`pitcher_throws` are
static, pre-game-knowable attributes (not outcomes), so matchups.py reads
them directly off the target game's own `pa` rows instead of rolling them.

---

## `matchups` (`data/processed/matchups/*.parquet`)

Two rows per `game_pk` (see the architecture note in the pipeline spec on
why this isn't the duplicated-row/complementary-label mistake it resembles)
— **both rows must always be routed to the same split; see
`datasets.chronological_split`.**

| column | meaning | window |
|---|---|---|
| `game_pk`, `game_date`, `season`, `park_id` | game identity | n/a |
| `batting_team`, `pitching_team`, `batting_is_home` | which side of this row is batting | n/a |
| `is_doubleheader_game_2` | see `pa` table | n/a |
| `runs_scored` | **label** — this row's batting team's final score | end of game |
| `lineup_player_ids` | 9 batter ids, ordered by starting slot | pre-game (starting lineup only — see v1 caveat) |
| `slot{1-9}_batter_id`, `slot{1-9}_stands` | that slot's starter + handedness | pre-game |
| `slot{1-9}_expected_pa` | **empirical**, leakage-safe (see `matchups.compute_slot_pa_table`): the running mean PAs received by that slot, through the end of the prior calendar date, for that season. Falls back to `config.DEFAULT_PA_PER_SLOT[slot]` until `config.SLOT_PA_MIN_TEAM_GAMES` of same-season sample exists. This is the feature that breaks the lineup's rotational symmetry — deliberately *measured*, not a config constant, since it varies by season/run-environment | leakage-safe running mean, current season only, reset at each season's start |
| `slot{1-9}_{feature}` for every `features.HITTER_FEATURE_COLUMNS` | that slot's starter's pre-game hitter features | as of the day before this game (see `player_daily`) |
| `sp_id`, `sp_throws` | starting pitcher + handedness | pre-game |
| `sp_{feature}` for every `features.PITCHER_FEATURE_COLUMNS` | starter's pre-game pitcher features | as of the day before this game |
| `bullpen_pitcher_ids`, `n_candidates` | the (up to `config.MAX_BULLPEN_SIZE`, truncated) reliever ids actually used to fill `bullpen{k}_*` below, and how many were available before truncation (traceability / diagnostic — `n_candidates > MAX_BULLPEN_SIZE` means this row got truncated) | as of the day before this game |
| `bullpen{k}_pitcher_id`, `bullpen{k}_mask`, `bullpen{k}_ip_share`, `bullpen{k}_throws`, `bullpen{k}_{feature}` for every `PITCHER_FEATURE_COLUMNS`, for k in 1..`config.MAX_BULLPEN_SIZE` | one **individual** reliever's own pre-game features per slot — a set, not a composite (see below) — ordered most- to least-recently-used, padded with `mask=False` when fewer than `MAX_BULLPEN_SIZE` are available. Padding rows get finite `0.0` placeholders, never `NaN` | as of the day before this game |

**Why a set of individual pitchers, not one composite vector:** an earlier
version of this table collapsed the bullpen into a single IP-share-weighted
average vector. `PitcherEncoder` (the model consuming this table) turned out
to expect per-pitcher granularity with an explicit mask and per-slot
weights, so `resolve_bullpen_set` now resolves each candidate individually.
`datasets.py` stacks `sp_*` (always index 0) with `bullpen{1..K}_*` into the
`(N, K, F_p)` tensor the encoder actually reads.

**Bullpen availability (v1 heuristic, still a documented limitation):** a
pitcher counts as "on the bullpen" if he pitched for this team at least once
in the trailing `config.BULLPEN_ROSTER_LOOKBACK_DAYS` days (a recent-
activity proxy for roster membership — misses an IL activation with no
recent appearances, and would double-count a recently-traded player who
pitched for both clubs within the window); he's then "available" if his
pitch counts over the trailing 1/2 days are under
`config.BULLPEN_MAX_PITCHES_1D`/`_2D`. Candidates beyond `MAX_BULLPEN_SIZE`
are truncated, keeping the most recently used (a deterministic tiebreak,
not a claim of significance — see `bullpen_candidates`'s truncation-rate log
line). If literally no reliever qualifies (observed only on a season's very
first game, before any relief appearance exists in the ingested window),
every `bullpen{k}_mask` is `False` and features are `0.0` placeholders.

---

## Tensors (`datasets.build_tensors`)

| tensor | shape | contents |
|---|---|---|
| `lineup` | `(N, 9, F_h)` | per slot: `HITTER_FEATURE_COLUMNS` + `expected_pa` + handedness (`L`→-1, `R`→1, `S`→0) |
| `pa_weights` | `(N, 9)` | the **same** `expected_pa` values, exposed as their own raw (never scaled) tensor — `HitterEncoder`'s `pooling="pa_weighted"` needs real PA counts to average over, not z-scored ones |
| `pitchers` | `(N, K, F_p)`, `K = 1 + config.MAX_BULLPEN_SIZE` | index 0 = starter (`sp_*` + handedness); 1..K-1 = bullpen slots (`bullpen{k}_*` + handedness) |
| `pitcher_mask` | `(N, K)` bool | `True` at every real pitcher, `False` at padding. Index 0 is always `True` (asserted in `build_pitcher_set`) |
| `pitcher_weights` | `(N, K)` | expected innings per slot, RAW (never scaled): `config.SP_EXPECTED_IP` at index 0, each slot's `ip_share` elsewhere (`0.0` at padding) |
| `context` | `(N, F_c)` | `batting_is_home`, `park_code` (categorical index via `build_park_vocab`), `month`, `sp_is_cold_start`, `lineup_cold_start_frac` (mean over 9 slots), `bullpen_cold_start_frac` (mean over only the *real*, unpadded bullpen slots; `1.0` if none are real) |
| `y` | `(N,)` | `runs_scored` |
| `game_pk` | `(N,)` | carried through for group-aware splitting, never fed to the model |

**Splitting:** `chronological_split(seasons_train, season_val, season_test)`
assigns whole seasons to whole splits — no random splits anywhere. This
trivially satisfies both leakage invariants (a `game_pk` belongs to exactly
one season; consecutive seasons' date ranges don't overlap).

**Scaling:** `Scaler` is fit only on the training split's tensors (pooled
over the game axis, and additionally over the slot axis for `lineup`) and
persisted (`fit_and_persist_split`) alongside the split's `game_pk` lists,
so val/test are always scored against train-only statistics. `pitchers` is
fit **mask-aware**: only real (unmasked) rows contribute to the mean/std,
so padding's fixed `0.0` placeholder never pulls every stat toward zero.
`pa_weights`, `pitcher_mask`, `pitcher_weights`, `y`, and `game_pk` are never
scaled — see `datasets.SCALED_KEYS`.

**Loading for training:** `basenet.loader.load_split` wraps
`fit_and_persist_split` + `build_tensors` + `Scaler.transform` and returns
everything as device-placed PyTorch tensors, fully in memory (no
`Dataset`/`DataLoader` — the whole table is small enough to load once and
index directly).
