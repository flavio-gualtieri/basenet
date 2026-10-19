"""Synthetic raw-statcast-shaped fixtures + a temp BASENET_DATA_ROOT so the
full pipeline (pa -> features -> matchups -> datasets) can be exercised
end-to-end without network access or real Savant data.

The synthetic world is deliberately simple so its properties are checkable
by construction:
  - Two teams, AAA (frequently home) and BBB.
  - Each team bats a fixed 9-man lineup (ids 1-9 for AAA, 101-109 for BBB)
    that recurs across games, so rolling features actually accumulate.
  - Each team's half of a game is 2 "trips through the order" (18 PAs): the
    team's starter throws the first 9, a reliever throws the last 9 -- this
    gives the reliever a real player_daily row so bullpen-availability logic
    has something to find.
  - Runs only ever score on HR (every other outcome is a no-runners-score
    event) -- so the final box score for a team-game is just its HR count,
    making the runs-reconciliation invariant trivial to hand-verify in
    assertions.
  - One doubleheader pair (same date, same matchup, two game_pks) is
    included specifically to exercise `is_doubleheader_game_2` and the
    (game_date, game_pk) ordering key.
  - Two seasons (2021, 2022) so trailing-window continuity across a season
    boundary, and season-reset of the season-to-date window, are both
    exercised.
  - Every simulated game ends tied 2-2 (both teams run the same outcome
    cycle), so game 1003 gets one extra non-PA pitch: a walk-off wild pitch
    (`events` null) that makes it 3-2 home. That's the case the old
    terminal-pitch-only label got wrong (it recorded a tie).
  - A StatsAPI-shaped schedule snapshot is written alongside the raw
    statcast, with official scores, so pa.py's label cross-check runs.
"""
from __future__ import annotations

import datetime as dt
import itertools

import pandas as pd
import pytest

AAA_BATTERS = list(range(1, 10))
BBB_BATTERS = list(range(101, 110))
AAA_SP, AAA_RP = 901, 911
BBB_SP, BBB_RP = 902, 912

# 18 outcomes (2 trips through a 9-slot order), cycled per team per game.
OUTCOME_CYCLE = [
    "K", "BB", "1B", "OUT_IP", "HR", "2B", "OUT_IP", "HBP", "3B",
    "ROE", "K", "1B", "OUT_IP", "BB", "HR", "OUT_IP", "1B", "K",
]
BIP_OUTCOMES = {"OUT_IP", "1B", "2B", "3B", "HR", "ROE"}
# alternate ground_ball/fly_ball so GB%/FB% have both to work with
_BB_TYPE_CYCLE = itertools.cycle(["ground_ball", "fly_ball"])

DESCRIPTION_FOR_OUTCOME = {
    "K": "swinging_strike",
    "BB": "ball",
    "HBP": "hit_by_pitch",
}


def _description_and_bb_type(outcome: str) -> tuple[str, str | None]:
    if outcome in BIP_OUTCOMES:
        return "hit_into_play", next(_BB_TYPE_CYCLE)
    return DESCRIPTION_FOR_OUTCOME[outcome], None


def _simulate_game(
    game_pk: int, game_date: dt.date, season: int, home_team: str, away_team: str,
    home_batters: list[int], away_batters: list[int],
    home_sp: int, home_rp: int, away_sp: int, away_rp: int,
    at_bat_number_start: int = 0,
) -> tuple[list[dict], int]:
    """One game: 2 innings, top+bottom, 9 PAs/half/inning -> 18 PAs/team.
    Returns (rows, next_at_bat_number_start) for chaining doubleheader games.
    """
    rows = []
    home_score = away_score = 0
    at_bat_number = at_bat_number_start
    team_pa_counters = {home_team: 0, away_team: 0}

    for inning in (1, 2):
        for half in ("Top", "Bot"):
            batting_team = away_team if half == "Top" else home_team
            pitching_team = home_team if half == "Top" else away_team
            lineup = away_batters if half == "Top" else home_batters
            # first 9 PAs of the *team's pitching* -> starter; PAs 10-18 -> reliever
            starter = home_sp if pitching_team == home_team else away_sp
            reliever = home_rp if pitching_team == home_team else away_rp
            pitcher = starter if inning == 1 else reliever

            batter_hand_cycle = itertools.cycle(["R", "L"])
            for i in range(9):
                at_bat_number += 1
                team_pa_counters[batting_team] += 1
                slot_idx = (team_pa_counters[batting_team] - 1) % 9
                batter = lineup[slot_idx]
                outcome = OUTCOME_CYCLE[(team_pa_counters[batting_team] - 1) % len(OUTCOME_CYCLE)]
                description, bb_type = _description_and_bb_type(outcome)

                pre_home, pre_away = home_score, away_score
                if outcome == "HR":
                    if batting_team == home_team:
                        home_score += 1
                    else:
                        away_score += 1
                bat_score = pre_home if batting_team == home_team else pre_away
                post_bat_score = home_score if batting_team == home_team else away_score

                # Each pitcher (starter in inning 1, reliever in inning 2) only
                # ever faces exactly one 9-batter trip through the lineup in
                # this synthetic game, so it's always his first time through.
                n_thru = 1

                rows.append(dict(
                    game_pk=game_pk, game_date=game_date, season=season, game_type="R",
                    home_team=home_team, away_team=away_team,
                    inning=inning, inning_topbot=half,
                    at_bat_number=at_bat_number,
                    batter=batter, pitcher=pitcher,
                    stand=next(batter_hand_cycle), p_throws="R" if pitcher in (home_sp, away_sp) else "L",
                    n_thruorder_pitcher=n_thru,
                    bat_score=bat_score, post_bat_score=post_bat_score,
                    home_score=pre_home, away_score=pre_away,
                    post_home_score=home_score, post_away_score=away_score,
                    events=outcome_to_event(outcome), bb_type=bb_type, description=description,
                ))
    return rows, at_bat_number


_EVENT_FOR_OUTCOME = {
    "K": "strikeout", "BB": "walk", "HBP": "hit_by_pitch",
    "OUT_IP": "field_out", "1B": "single", "2B": "double", "3B": "triple",
    "HR": "home_run", "ROE": "field_error",
}


def outcome_to_event(outcome: str) -> str:
    return _EVENT_FOR_OUTCOME[outcome]


def build_synthetic_raw() -> pd.DataFrame:
    rows: list[dict] = []

    # season 2021: 4 games, including a same-day doubleheader (game_pk 1000, 1001)
    games_2021 = [
        (1000, dt.date(2021, 4, 1), "AAA", "BBB"),
        (1001, dt.date(2021, 4, 1), "AAA", "BBB"),  # doubleheader game 2
        (1002, dt.date(2021, 4, 3), "BBB", "AAA"),
        (1003, dt.date(2021, 4, 5), "AAA", "BBB"),
    ]
    # season 2022: 3 games (early-season games here exercise cross-season trailing)
    games_2022 = [
        (2000, dt.date(2022, 4, 1), "AAA", "BBB"),
        (2001, dt.date(2022, 4, 3), "BBB", "AAA"),
        (2002, dt.date(2022, 4, 5), "AAA", "BBB"),
    ]
    # season 2023: small, exists only so tests can exercise a real 3-way
    # chronological split (train=2021, val=2022, test=2023).
    games_2023 = [
        (3000, dt.date(2023, 4, 1), "AAA", "BBB"),
        (3001, dt.date(2023, 4, 3), "BBB", "AAA"),
    ]

    for season, games in ((2021, games_2021), (2022, games_2022), (2023, games_2023)):
        for game_pk, game_date, home_team, away_team in games:
            home_batters = AAA_BATTERS if home_team == "AAA" else BBB_BATTERS
            away_batters = AAA_BATTERS if away_team == "AAA" else BBB_BATTERS
            home_sp, home_rp = (AAA_SP, AAA_RP) if home_team == "AAA" else (BBB_SP, BBB_RP)
            away_sp, away_rp = (AAA_SP, AAA_RP) if away_team == "AAA" else (BBB_SP, BBB_RP)
            game_rows, _ = _simulate_game(
                game_pk, game_date, season, home_team, away_team,
                home_batters, away_batters, home_sp, home_rp, away_sp, away_rp,
            )
            rows.extend(game_rows)

    _append_non_pa_walkoff(rows, NON_PA_WALKOFF_GAME_PK)

    df = pd.DataFrame(rows)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


NON_PA_WALKOFF_GAME_PK = 1003


def _append_non_pa_walkoff(rows: list[dict], game_pk: int) -> None:
    """Bottom of the last inning, tied: the home team's next batter steps in
    and the winning run scores on a wild pitch before the PA ends. The game
    is over, so this pitch carries no `events` value and no PA row exists.
    """
    last = [r for r in rows if r["game_pk"] == game_pk][-1]
    assert last["inning_topbot"] == "Bot" and last["post_home_score"] == last["post_away_score"]
    rows.append(dict(
        last,
        at_bat_number=last["at_bat_number"] + 1,
        bat_score=last["post_home_score"], post_bat_score=last["post_home_score"] + 1,
        home_score=last["post_home_score"], away_score=last["post_away_score"],
        post_home_score=last["post_home_score"] + 1, post_away_score=last["post_away_score"],
        events=None, bb_type=None, description="ball",
    ))


def build_synthetic_schedule(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    """Official-score schedule in ingest.SCHEDULE_COLUMNS shape, derived
    independently of pa.py (max score over all pitches = the true final in
    this synthetic world)."""
    from basenet.data.ingest import SCHEDULE_COLUMNS
    g = raw[raw["season"] == season].groupby("game_pk").agg(
        official_date=("game_date", "first"), home_team_name=("home_team", "first"),
        away_team_name=("away_team", "first"),
        home_score=("post_home_score", "max"), away_score=("post_away_score", "max"),
    ).reset_index()
    g["season"] = season
    g["game_type"] = "R"
    g["coded_game_state"] = "F"
    g["detailed_state"] = "Final"
    g["home_team_id"] = g["home_team_name"].map({"AAA": 1, "BBB": 2})
    g["away_team_id"] = g["away_team_name"].map({"AAA": 1, "BBB": 2})
    g["venue_id"] = g["home_team_id"] + 100
    g["venue_name"] = g["home_team_name"] + " Park"
    g["double_header"] = "N"
    g["game_number"] = 1
    for c in ("home_score", "away_score", "venue_id"):
        g[c] = g[c].astype("Int64")
    return g[SCHEDULE_COLUMNS]


@pytest.fixture(scope="session")
def synthetic_raw_df() -> pd.DataFrame:
    return build_synthetic_raw()


SEASONS = (2021, 2022, 2023)


@pytest.fixture()
def pipeline_env(tmp_path, monkeypatch, synthetic_raw_df):
    """Points BASENET_DATA_ROOT at a scratch dir and writes the synthetic raw
    snapshot in the layout ingest.py would have produced, for seasons 2021-2023.
    Returns the data root path. Runs pa/features/matchups for real.
    """
    monkeypatch.setenv("BASENET_DATA_ROOT", str(tmp_path))
    from basenet.data import paths, pa, features, matchups

    for season in SEASONS:
        season_df = synthetic_raw_df[synthetic_raw_df["season"] == season]
        out_dir = paths.raw_statcast_season_dir(season)
        out_dir.mkdir(parents=True, exist_ok=True)
        season_df.to_parquet(out_dir / f"statcast_{season}.parquet", index=False)
        sched_dir = paths.raw_schedule_season_dir(season)
        sched_dir.mkdir(parents=True, exist_ok=True)
        build_synthetic_schedule(synthetic_raw_df, season).to_parquet(
            sched_dir / f"schedule_{season}.parquet", index=False
        )

    for season in SEASONS:
        df = pa.build_pa_table(season)
        out_dir = paths.interim_pa_season_dir(season)
        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_dir / f"pa_{season}.parquet", index=False)

    player_daily = features.build_player_daily(list(SEASONS))
    paths.INTERIM_PLAYER_DAILY_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        player_daily[player_daily["season"] == season].to_parquet(
            paths.INTERIM_PLAYER_DAILY_DIR / f"player_daily_{season}.parquet", index=False
        )
    post_game = features.build_pitcher_post_game(list(SEASONS))
    paths.INTERIM_PITCHER_POST_GAME_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        post_game[post_game["season"] == season].to_parquet(
            paths.INTERIM_PITCHER_POST_GAME_DIR / f"pitcher_post_game_{season}.parquet", index=False
        )

    matchups_df = matchups.build_matchups(list(SEASONS))
    paths.PROCESSED_MATCHUPS_DIR.mkdir(parents=True, exist_ok=True)
    for season in SEASONS:
        matchups_df[matchups_df["season"] == season].to_parquet(
            paths.PROCESSED_MATCHUPS_DIR / f"matchups_{season}.parquet", index=False
        )

    return tmp_path
