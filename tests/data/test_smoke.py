def test_pipeline_runs_end_to_end(pipeline_env):
    from basenet.data import paths
    matchups_2021 = paths.PROCESSED_MATCHUPS_DIR / "matchups_2021.parquet"
    assert matchups_2021.exists()
