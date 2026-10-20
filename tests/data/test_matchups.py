from __future__ import annotations

import datetime as dt

import pandas as pd

from basenet.data import config, matchups, paths


def _matchups_2021(pipeline_env) -> pd.DataFrame:
    return pd.read_parquet(paths.PROCESSED_MATCHUPS_DIR / "matchups_2021.parquet")


def test_starting_lineup_matches_synthetic_roster(pipeline_env):
    df = _matchups_2021(pipeline_env)
    row = df[(df["game_pk"] == 1000) & (df["batting_team"] == "AAA")].iloc[0]
    lineup = list(row["lineup_player_ids"])
    assert lineup == list(range(1, 10))  # AAA's fixed 1-9 roster, in slot order


def test_starting_pitcher_resolves_to_correct_team(pipeline_env):
    df = _matchups_2021(pipeline_env)
    # game 1000: AAA home, BBB away -> when AAA bats, BBB's starter (902) pitches
    row = df[(df["game_pk"] == 1000) & (df["batting_team"] == "AAA")].iloc[0]
    assert row["sp_id"] == 902
    row2 = df[(df["game_pk"] == 1000) & (df["batting_team"] == "BBB")].iloc[0]
    assert row2["sp_id"] == 901


def test_bullpen_set_excludes_starter_and_finds_reliever(pipeline_env):
    df = _matchups_2021(pipeline_env)
    # game 1002 (2021-04-03, third game overall): by now both teams' relievers
    # (911, 912) have a prior relief outing on the books, so they should be
    # eligible bullpen candidates; the starting pitcher must never appear
    # in his own team's bullpen list.
    row = df[(df["game_pk"] == 1002) & (df["batting_team"] == "BBB")].iloc[0]
    assert row["sp_id"] not in row["bullpen_pitcher_ids"]
    assert len(row["bullpen_pitcher_ids"]) > 0
    # slot 1 is real (mask=True): reliever 911/912 pitched here, features resolve
    assert row["bullpen1_mask"]
    assert not pd.isna(row["bullpen1_std_k_pct"])
    # slots beyond however many real relievers exist are padding
    n_real = len(row["bullpen_pitcher_ids"])
    from basenet.data import config
    for k in range(n_real + 1, config.MAX_BULLPEN_SIZE + 1):
        assert not row[f"bullpen{k}_mask"]
        assert row[f"bullpen{k}_ip_share"] == 0.0


def test_compute_slot_pa_table_leakage_safe(monkeypatch):
    """Standalone (no pipeline_env needed): expected_pa for a given date must
    only reflect team-games strictly before it, and must fall back to the
    static default before any same-season sample exists at all.
    """
    monkeypatch.setattr(config, "SLOT_PA_MIN_TEAM_GAMES", 2)
    # slot 1: team A gets 4 PAs on day1, 5 on day2, 6 on day3 (all season 2099)
    rows = []
    for game_pk, date, pa_count in [(1, dt.date(2099, 4, 1), 4),
                                     (2, dt.date(2099, 4, 2), 5),
                                     (3, dt.date(2099, 4, 3), 6)]:
        for _ in range(pa_count):
            rows.append({"game_pk": game_pk, "batting_team": "A", "order_slot": 1,
                         "game_date": date, "season": 2099})
    pa_df = pd.DataFrame(rows)
    pa_df["game_date"] = pd.to_datetime(pa_df["game_date"])

    table = matchups.compute_slot_pa_table(pa_df)
    by_date = table.set_index("game_date")["expected_pa"]

    # day 1: zero prior team-games -> bootstrap fallback, not 4 (its own count)
    assert by_date[pd.Timestamp("2099-04-01")] == config.DEFAULT_PA_PER_SLOT[1]
    # day 2: exactly 1 prior team-game (day 1's 4 PAs) -- still below the
    # monkeypatched min of 2 -> still the fallback, not day 1's own 4.
    assert by_date[pd.Timestamp("2099-04-02")] == config.DEFAULT_PA_PER_SLOT[1]
    # day 3: 2 prior team-games (4 + 5 PAs over 2 games) -> real average 4.5,
    # and critically NOT influenced by day 3's own 6 PAs.
    assert by_date[pd.Timestamp("2099-04-03")] == 4.5


def test_two_rows_have_swapped_home_away(pipeline_env):
    df = _matchups_2021(pipeline_env)
    g = df[df["game_pk"] == 1000]
    assert set(g["batting_team"]) == {"AAA", "BBB"}
    assert set(g["pitching_team"]) == {"AAA", "BBB"}
    home_row = g[g["batting_is_home"]].iloc[0]
    away_row = g[~g["batting_is_home"]].iloc[0]
    assert home_row["batting_team"] == home_row["park_id"]
    assert away_row["batting_team"] != away_row["park_id"]


def test_bullpen_features_describe_reliever_going_into_target_game(pipeline_env):
    """Game 1002 (2021-04-03): AAA's reliever 911 last pitched in BOTH games
    of the 2021-04-01 doubleheader (1000, 1001), 9 batters / 9 pitches each.
    His bullpen features for 1002 must reflect both outings -- not his state
    before his last outing (the old as-of bug: std_bf 9, days_rest 0).
    """
    df = _matchups_2021(pipeline_env)
    row = df[(df["game_pk"] == 1002) & (df["pitching_team"] == "AAA")].iloc[0]
    k = list(row["bullpen_pitcher_ids"]).index(911) + 1
    assert row[f"bullpen{k}_std_bf"] == 18           # through game 1001, inclusive
    assert row[f"bullpen{k}_days_rest"] == 2         # 04-03 minus 04-01
    assert row[f"bullpen{k}_pitch_count_last_outing"] == 9
    assert row[f"bullpen{k}_pitch_count_trailing_3d"] == 18
    assert not row[f"bullpen{k}_is_cold_start"]


def test_bullpen_excludes_rotation_starters_keeps_swingman():
    """Standalone: team T starts A, then B, then C (target game). R relieves
    every game; S (swingman) relieves twice and starts once. A and B are
    rotation starters -- not bullpen candidates for C's game -- R and S are.
    """
    games = [(1, dt.date(2099, 5, 1), [("A", 1), ("R", 2), ("S", 3)]),
             (2, dt.date(2099, 5, 3), [("B", 1), ("R", 2), ("S", 3)]),
             (3, dt.date(2099, 5, 5), [("S", 1), ("R", 2)]),
             (4, dt.date(2099, 5, 9), [("C", 1)])]
    ids = {"A": 1, "B": 2, "C": 3, "R": 4, "S": 5}
    pa_rows, pc_rows = [], []
    for game_pk, date, appearances in games:
        for name, ab in appearances:
            pa_rows.append({"game_pk": game_pk, "pitching_team": "T", "pitcher_id": ids[name],
                            "pitcher_throws": "R", "game_date": pd.Timestamp(date), "at_bat_number": ab})
            pc_rows.append({"pitcher_id": ids[name], "game_pk": game_pk, "game_date": pd.Timestamp(date),
                            "n_pitches": 10, "pitching_team": "T"})
    pa_df, pc_df = pd.DataFrame(pa_rows), pd.DataFrame(pc_rows)
    target = pd.DataFrame({"game_pk": [4], "pitching_team": ["T"], "game_date": [pd.Timestamp(2099, 5, 9)]})
    out = matchups.bullpen_candidates(pa_df, pc_df, target)
    assert set(out.iloc[0]["bullpen_ids"]) == {ids["R"], ids["S"]}


def test_bullpen_order_and_truncation_ignore_input_row_order():
    """Standalone: 16 relievers all last pitched on 05-01, one more (late)
    on 05-03 -- more than MAX_BULLPEN_SIZE, mostly tied on recency. Most
    recent first, ties by id, so the kept set and its order must be the same
    however the input rows happen to be ordered (DuckDB doesn't promise a
    row order). MLBAM-sized ids, whose set iteration order isn't numeric."""
    tied = [650000 + 97 * k for k in range(16)]
    late = 700001
    games = [(1, dt.date(2099, 5, 1), [1] + tied[::-1]),
             (2, dt.date(2099, 5, 3), [1, late]),
             (3, dt.date(2099, 5, 9), [2])]
    pa_rows, pc_rows = [], []
    for game_pk, date, pitcher_ids in games:
        for ab, pid in enumerate(pitcher_ids, start=1):
            pa_rows.append({"game_pk": game_pk, "pitching_team": "T", "pitcher_id": pid,
                            "pitcher_throws": "R", "game_date": pd.Timestamp(date), "at_bat_number": ab})
            pc_rows.append({"pitcher_id": pid, "game_pk": game_pk, "game_date": pd.Timestamp(date),
                            "n_pitches": 10, "pitching_team": "T"})
    pa_df, pc_df = pd.DataFrame(pa_rows), pd.DataFrame(pc_rows)
    target = pd.DataFrame({"game_pk": [3], "pitching_team": ["T"], "game_date": [pd.Timestamp(2099, 5, 9)]})
    expected = [late] + sorted(tied)[: config.MAX_BULLPEN_SIZE - 1]
    for seed in range(5):
        shuffled = pc_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        out = matchups.bullpen_candidates(pa_df, shuffled, target)
        assert list(out.iloc[0]["bullpen_ids"]) == expected
        assert out.iloc[0]["n_candidates"] == 17


def test_venue_id_comes_from_schedule(pipeline_env):
    """venue_id is StatsAPI's real venue (synthetic: home team id + 100),
    carried on both rows of every game; park_id stays the home-team code."""
    df = _matchups_2021(pipeline_env)
    expected = df["park_id"].map({"AAA": 101, "BBB": 102})
    assert (df["venue_id"] == expected).all()
