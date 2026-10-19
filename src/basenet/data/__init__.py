"""MLB run-prediction data pipeline: ingest -> pa -> features -> matchups -> datasets.

See DATA.md at the repo root for the column-by-column contract of the
`matchups` table, and each module's docstring for its stage's leakage
contract.
"""
