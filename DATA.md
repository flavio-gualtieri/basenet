# DATA.md — the `matchups` training table

This documents every column produced by the data pipeline (`src/basenet/data/`),
with special attention to **the exact time window each feature is computed
over** — that's the part that determines whether the table is safe to train
on. See each module's docstring for the mechanics; this file is the contract.

Pipeline: `ingest -> pa -> features -> matchups -> datasets`. Run order and
CLI: `make data` (see README.md) / each module's `--help`.

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
- **`park_id` is the home team's code; `venue_id` is the real ballpark.**
  Statcast carries no venue column, so `park_id` (used to tell the home side
  from the away side) is a team code. The actual ballpark is `venue_id`,
  MLB StatsAPI's venue id per `game_pk` from the schedule snapshot — e.g. it
  separates the Rays' and A's 2025 temporary homes (Steinbrenner Field,
  Sutter Health Park) and the London/Mexico City/Seoul/Tokyo series from
  their usual parks. The model's park embedding is keyed on `venue_id`.
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
| `venue_id` | StatsAPI venue id — the ballpark actually played in | n/a (known pre-game) |
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
| `woba_value`, `woba_denom` | statcast's wOBA run value of the PA and whether it counts toward wOBA (excludes sac bunts, catcher's interference, ...) | this PA |
| `launch_speed`, `launch_speed_angle`, `xwoba_contact` | exit velocity (mph), statcast's contact-quality code (`6` = barrel), and expected wOBA from exit velocity + launch angle (`estimated_woba_using_speedangle`) — balls in play only | this PA |
| `at_bat_number` | internal ordering aid, not a model feature | n/a |
| `final_home_score`, `final_away_score` | **official** final score from MLB StatsAPI (`data/raw/schedule/`), carried on every row of the game. Statcast's own final is cross-checked against it and `pa` fails on any disagreement except a walk-off with no pitch thrown (e.g. a balk), which statcast cannot see | end of game (**label — never used as a pre-game feature**) |

**Label leakage note:** `final_home_score`/`final_away_score` are the
label source (official scores, see above) and also let `pa.reconcile_runs`
check "do the summed `runs_on_play` equal the final score". The only
tolerated gap is a home walk-off whose winning run scored on no plate
appearance (wild pitch, passed ball, balk): exactly one run, from a tie.
`validate.py` re-checks every `matchups.runs_scored` label against the
StatsAPI schedule.
They are outcome columns for the *current* game and must never be joined
into a feature for the model — matchups.py only ever reads them to compute
the label (`runs_scored`), never a feature.

---

## `player_daily` (`data/interim/player_daily/*.parquet`)

Keyed on **(`player_id`, `role`, `game_date`, `game_pk`)** — `role` is added
beyond the natural 3-part key because two-way players (a pitcher who
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
| `std_{stat}`, `trailing_{stat}` for `stat` in: hitters — `k_pct, bb_pct, hbp_pct, x1b_rate, x2b_rate, x3b_rate, hr_rate, babip, gb_pct, fb_pct`; pitchers — `k_pct, bb_pct, hr_per_bf, gb_pct, contact_pct, whiff_pct`; both (quality of contact, produced / allowed — the `contact` feature group) — `xwoba` (expected wOBA on contact, actual wOBA value otherwise, per wOBA PA), `hard_hit_pct` (≥ `config.HARD_HIT_MPH`), `barrel_pct` (per batted ball with a tracked exit velocity) | empirical-Bayes-shrunk rate: `(observed_count + k*league_rate) / (n + k)`, `k` from `config.HITTER_SHRINKAGE_K` / `PITCHER_SHRINKAGE_K` | same STD / trailing window as the count above; `league_rate` is the running league mean for that stat/role/season through the **end of the prior calendar date** (see "league mean" note below) |
| `is_cold_start` | `trailing_pa == 0` (hitter) / `trailing_bf == 0` (pitcher) — literal debut in the ingested window (every 2021 player's first game, since the window starts there; ~1% of starters from 2022 on). A season opener for a returning player is *not* a cold start — it shows up as `std_pa`/`std_bf == 0` | n/a (flag, not a rate) |
| `days_since_last_game` (hitter) / `days_rest` (pitcher) | calendar days since the player's previous game/outing; default (`config.DEFAULT_DAYS_SINCE_LAST_GAME` / `DEFAULT_DAYS_REST`) on a debut | up to and including the prior appearance |
| `pitch_count_last_outing` (pitcher) | total pitches thrown in the immediately preceding outing | prior outing only |
| `pitch_count_trailing_3d` (pitcher) | total pitches thrown in the trailing 3 calendar days | strictly before this game |
| `plat_vs_l_{stat}`, `plat_vs_r_{stat}` — hitters: `k_pct, bb_pct, hr_rate, xwoba` vs LHP / RHP; pitchers: `k_pct, bb_pct, hr_per_bf, xwoba` vs LHB / RHB (`platoon` group) | the rate vs that hand, shrunk toward the player's own overall rate over the same window with `config.PLATOON_SHRINKAGE_K`; that overall rate is itself shrunk toward the running league rate with the stat's usual `k`. No PAs vs a hand → his overall rate; no PAs at all → the league rate | every PA strictly before this game in the **whole ingested window** (splits need the sample; not season-reset) |
| `n_recent_starts`, `bf_per_start` (pitcher; `workload` group) | batters faced per start over his last `config.STARTER_BF_LOOKBACK_STARTS` starts, shrunk toward `config.STARTER_BF_PRIOR` with `config.STARTER_BF_PRIOR_K` starts. A start = the first pitcher his team used (openers count, and pull it down) | starts strictly before this game, any season |

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

## `pitcher_post_game` (`data/interim/pitcher_post_game/*.parquet`)

Keyed on (`player_id`, `game_date`, `game_pk`), one row per pitcher-game,
with `features.PITCHER_STATE_COLUMNS` (the `player_daily` pitcher columns
minus the three fatigue features). Same windows as `player_daily`, but read
**after** the game (inclusive of it) rather than before. Only used to
resolve bullpen candidates (see the `matchups` notes); never joined onto the
game it describes.

---

## `matchups` (`data/processed/matchups/*.parquet`)

Two rows per `game_pk` (see the "Row identity" note in `matchups.py`'s docstring on
why this isn't the duplicated-row/complementary-label mistake it resembles)
— **both rows must always be routed to the same split; see
`datasets.chronological_split`.**

| column | meaning | window |
|---|---|---|
| `game_pk`, `game_date`, `season`, `park_id`, `venue_id` | game identity (see conventions on `park_id` vs `venue_id`) | n/a |
| `batting_team`, `pitching_team`, `batting_is_home` | which side of this row is batting | n/a |
| `is_doubleheader_game_2` | see `pa` table | n/a |
| `runs_scored` | **label** — this row's batting team's final score | end of game |
| `lineup_player_ids` | 9 batter ids, ordered by starting slot | pre-game (starting lineup only — see v1 caveat) |
| `slot{1-9}_batter_id`, `slot{1-9}_stands` | that slot's starter + handedness | pre-game |
| `slot{1-9}_expected_pa` | **empirical**, leakage-safe (see `matchups.compute_slot_pa_table`): the running mean PAs received by that slot, through the end of the prior calendar date, for that season. Falls back to `config.DEFAULT_PA_PER_SLOT[slot]` until `config.SLOT_PA_MIN_TEAM_GAMES` of same-season sample exists. This is the feature that breaks the lineup's rotational symmetry — deliberately *measured*, not a config constant, since it varies by season/run-environment | leakage-safe running mean, current season only, reset at each season's start |
| `slot{1-9}_{feature}` for every `features.HITTER_FEATURE_COLUMNS` | that slot's starter's pre-game hitter features | as of the day before this game (see `player_daily`) |
| `sp_id`, `sp_throws` | starting pitcher + handedness | pre-game |
| `sp_{feature}` for every `features.PITCHER_FEATURE_COLUMNS` | starter's pre-game pitcher features | as of the day before this game |
| `sp_n_recent_starts`, `sp_bf_per_start` | starter's workload (see `player_daily`) | as of the day before this game |
| `slot{1-9}_plat_{stat}`, `slot{1-9}_sp_plat_{stat}` | resolved platoon matchup (`matchups.resolve_matchup_platoon`): the hitter's split vs **today's starter's hand**, and the starter's split vs **that hitter's side** (the side he batted from against this starter, so switch-hitters resolve correctly) | as of the day before this game |
| `is_day_game`, `bat_days_since_last_game` / `pit_days_since_last_game`, `bat_traveled` / `pit_traveled`, `bat_day_after_night` / `pit_day_after_night` (`schedule` group) | from the StatsAPI schedule: day vs night; each team's days since its previous game this season (0 = doubleheader game 2; capped at `config.TEAM_REST_DAYS_CAP`, also the opener value); whether it changed venue since then; day game the day after a night game. `bat_*` = the batting team, `pit_*` = its opponent | pre-game (the schedule) |
| `temp_f`, `wind_out_mph`, `wind_in_mph`, `is_indoor`, `is_precip` (`weather` group) | StatsAPI's official game-time weather: temperature (placeholder values below `config.MIN_PLAUSIBLE_TEMP_F` → `config.DEFAULT_TEMP_F`), wind speed blowing out / in (cross, calm, varying → 0; indoors → 0), dome or roof closed, drizzle/rain/snow | **first pitch** — the reported conditions, a slightly optimistic stand-in for the forecast a pre-game model would have (same caveat as actual lineups) |
| `bullpen_pitcher_ids`, `n_candidates` | the (up to `config.MAX_BULLPEN_SIZE`, truncated) reliever ids actually used to fill `bullpen{k}_*` below, and how many were available before truncation (traceability / diagnostic — `n_candidates > MAX_BULLPEN_SIZE` means this row got truncated) | as of the day before this game |
| `bullpen{k}_pitcher_id`, `bullpen{k}_mask`, `bullpen{k}_ip_share`, `bullpen{k}_throws`, `bullpen{k}_{feature}` for every `PITCHER_FEATURE_COLUMNS`, for k in 1..`config.MAX_BULLPEN_SIZE` | one **individual** reliever's own pre-game features per slot — a set, not a composite (see below) — ordered most- to least-recently-used, padded with `mask=False` when fewer than `MAX_BULLPEN_SIZE` are available. Padding rows get finite `0.0` placeholders, never `NaN` | as of the day before this game |

**Why a set of individual pitchers, not one composite vector:** an earlier
version of this table collapsed the bullpen into a single IP-share-weighted
average vector. `PitcherEncoder` (the model consuming this table) turned out
to expect per-pitcher granularity with an explicit mask and per-slot
weights, so `resolve_bullpen_set` now resolves each candidate individually.
`datasets.py` stacks `sp_*` (always index 0) with `bullpen{1..K}_*` into the
`(N, K, F_p)` tensor the encoder actually reads.

**Bullpen features describe the reliever going into the target game.**
A reliever usually didn't pitch in the target game, so there's no
`player_daily` row for him there. His state (counts, shrunk rates,
`is_cold_start`) comes from `pitcher_post_game` — his state *after* his
latest appearance strictly before the target game, which is his state going
into it. His fatigue features (`days_rest`, `pitch_count_last_outing`,
`pitch_count_trailing_3d`) are recomputed against the target game's date,
with the same definitions used for starters.

**Bullpen availability (v1 heuristic, still a documented limitation):** a
pitcher counts as "on the bullpen" if he pitched for this team at least once
in the trailing `config.BULLPEN_ROSTER_LOOKBACK_DAYS` days (a recent-
activity proxy for roster membership — misses an IL activation with no
recent appearances, and would double-count a recently-traded player who
pitched for both clubs within the window), and fewer than
`config.BULLPEN_STARTER_SHARE_EXCLUDE` of those appearances were starts
(drops the rotation; keeps openers and swingmen); he's then "available" if his
pitch counts over the trailing 1/2 days are under
`config.BULLPEN_MAX_PITCHES_1D`/`_2D`. Candidates beyond `MAX_BULLPEN_SIZE`
are truncated, keeping the most recently used, ties broken by pitcher id (a deterministic tiebreak,
not a claim of significance — see `bullpen_candidates`'s truncation-rate log
line). If literally no reliever qualifies (observed only on a season's very
first game, before any relief appearance exists in the ingested window),
every `bullpen{k}_mask` is `False` and features are `0.0` placeholders.

---

## Tensors (`datasets.build_tensors`)

**Feature groups.** The pipeline always computes every column above; the
optional groups in `datasets.FEATURE_GROUPS` (`contact`, `platoon`,
`workload`, `schedule`, `weather`) only choose which reach the tensors.
They're switched on in `configs.yaml` (`features.groups`), which also sets
the model's input widths (`datasets.feature_dims`), so a group can be tried
or dropped without rebuilding any data.

| tensor | shape | contents |
|---|---|---|
| `lineup` | `(N, 9, F_h)` | per slot: `datasets.hitter_columns(groups)` + `expected_pa` + handedness (`L`→-1, `R`→1, `S`→0) |
| `pa_weights` | `(N, 9)` | the **same** `expected_pa` values, exposed as their own raw (never scaled) tensor — `HitterEncoder`'s `pooling="pa_weighted"` needs real PA counts to average over, not z-scored ones |
| `pitchers` | `(N, K, F_p)`, `K = 1 + config.MAX_BULLPEN_SIZE` | index 0 = starter (`sp_*` + handedness); 1..K-1 = bullpen slots (`bullpen{k}_*` + handedness) |
| `pitcher_mask` | `(N, K)` bool | `True` at every real pitcher, `False` at padding. Index 0 is always `True` (asserted in `build_pitcher_set`) |
| `pitcher_weights` | `(N, K)` | expected innings per slot, RAW (never scaled): `config.SP_EXPECTED_IP` at index 0, each slot's `ip_share` elsewhere (`0.0` at padding). With the `workload` group: the starter's own expected innings (`sp_bf_per_start / config.BATTERS_PER_INNING`, kept within 1–8) at index 0, and the bullpen's shares rescaled to fill the rest of `config.TEAM_INNINGS_PER_GAME` |
| `context` | `(N, F_c)` | `batting_is_home`, `month`, `sp_is_cold_start`, `lineup_cold_start_frac` (mean over 9 slots), `bullpen_cold_start_frac` (mean over only the *real*, unpadded bullpen slots; `1.0` if none are real), then optional-group columns in `datasets.context_columns(groups)` order |
| `park` | `(N,)` int64 | index of `venue_id` in the train-fit vocab (`build_park_vocab`), for BaseNet's park embedding. `0` = unknown: a venue unseen in train, or with fewer than `datasets.PARK_MIN_TRAIN_ROWS` train rows (one-off venues). Never scaled — it's categorical |
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
`pa_weights`, `pitcher_mask`, `pitcher_weights`, `park`, `y`, and `game_pk`
are never scaled — see `datasets.SCALED_KEYS`. The park vocab is fit on the
train split only too, persisted with the scaler.

**Loading for training:** `basenet.loader.load_split` wraps
`fit_and_persist_split` + `build_tensors` + `Scaler.transform` and returns
everything as device-placed PyTorch tensors, fully in memory (no
`Dataset`/`DataLoader` — the whole table is small enough to load once and
index directly).
