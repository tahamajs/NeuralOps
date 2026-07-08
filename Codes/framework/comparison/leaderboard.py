"""Leaderboard generator."""

import pandas as pd


def generate_leaderboard(
    group_stats: pd.DataFrame,
    metric: str = "eval_accuracy",
    sort_by: str = "mean",
    ascending: bool = False,
) -> pd.DataFrame:
    df = group_stats.copy()
    df = df.sort_values(by=sort_by, ascending=ascending)
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1
    df.index.name = "Rank"
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    df["Medal"] = df.index.map(medals).fillna("")
    cols = ["group", "mean", "std", "ci95", "n", "Medal"]
    df = df[[c for c in cols if c in df.columns]]
    df.columns = ["Model / Method", "Mean", "Std", "CI95", "N", "Medal"][
        : len(df.columns)
    ]
    return df
