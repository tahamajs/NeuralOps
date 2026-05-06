"""Leaderboard generator for model comparison."""

import pandas as pd
from typing import Optional, Dict, Any

def generate_leaderboard(
    group_stats: pd.DataFrame,
    metric: str = "eval_accuracy",
    sort_by: str = "mean",
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Generate a formatted leaderboard from group statistics.

    Columns: Rank, Group, Mean, Std, CI95, N, Medal
    """
    df = group_stats.copy()
    df = df.sort_values(by=sort_by, ascending=ascending)
    df.reset_index(drop=True, inplace=True)
    df.index = df.index + 1   # 1‑based rank
    df.index.name = "Rank"

    # Add medal emojis for top 3
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    df["Medal"] = df.index.map(medals).fillna("")

    # Reorder columns
    cols = ["group", "mean", "std", "ci95", "n", "Medal"]
    df = df[[c for c in cols if c in df.columns]]
    df.columns = ["Model / Method", "Mean", "Std", "CI95", "N", "Medal"][:len(df.columns)]
    return df