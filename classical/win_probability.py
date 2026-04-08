"""
win_probability.py — Win Probability Model
============================================
Trains a logistic regression on historical NCAA tournament game outcomes
and produces a 64×64 win-probability matrix P[i][j] = P(team i beats team j).

Features used (all from ESPN team records):
  diff_ppg       points per game differential
  diff_papg      points allowed per game differential (flipped: lower is better)
  diff_win_pct   win percentage differential
  diff_pt_diff   average point differential

Public API
----------
build_training_features(games_df) -> pd.DataFrame
train_model(train_df)             -> (LogisticRegression, StandardScaler)
build_win_prob_matrix(teams, model, scaler) -> (np.ndarray, list[str])
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from data.fetch_stats import fetch_current_stats, fetch_historical_stats, fetch_historical_games

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_COLS = [
    "diff_ppg", "diff_papg", "diff_win_pct", "diff_pt_diff",
    "diff_recent_form", "diff_road_win_pct", "diff_margin_std", "diff_close_win_pct",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_diff(a, b) -> float:
    try:
        return float(a) - float(b)
    except (TypeError, ValueError):
        return 0.0


# ── Feature engineering ───────────────────────────────────────────────────────

def build_training_features(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    For every historical tournament game, fetch that season's team stats and
    compute (winner − loser) feature diffs. Each game produces two rows
    (both perspectives) so labels are balanced.

    Parameters
    ----------
    games_df : DataFrame with columns [season, winner_id, loser_id, ...]

    Returns
    -------
    DataFrame with columns [diff_ppg, diff_papg, diff_win_pct, diff_pt_diff,
                             diff_recent_form, diff_road_win_pct, diff_margin_std,
                             diff_close_win_pct, label]
    """
    from data.fetch_stats import fetch_team_schedule_features

    print("Building training features (fetching per-season stats)…")
    seasons = sorted(games_df["season"].unique())
    season_stats: dict[int, pd.DataFrame] = {}
    for s in seasons:
        df = fetch_historical_stats(s)
        if not df.empty:
            season_stats[s] = df.set_index("team_id")
        time.sleep(0.4)

    print("  Fetching schedule features for historical tournament teams…")
    sched_cache: dict[tuple, dict] = {}
    unique_pairs = set()
    for _, row in games_df.iterrows():
        s = int(row["season"])
        unique_pairs.add((str(row["winner_id"]), s))
        unique_pairs.add((str(row["loser_id"]), s))

    for i, (tid, s) in enumerate(sorted(unique_pairs)):
        sched_cache[(tid, s)] = fetch_team_schedule_features(tid, s)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(unique_pairs)} schedule fetches…")
        time.sleep(0.2)

    records: list[dict] = []
    skipped = 0
    for _, row in games_df.iterrows():
        s = int(row["season"])
        if s not in season_stats:
            skipped += 1
            continue
        stats  = season_stats[s]
        w_id   = str(row["winner_id"])
        l_id   = str(row["loser_id"])
        if w_id not in stats.index or l_id not in stats.index:
            skipped += 1
            continue
        w, l   = stats.loc[w_id], stats.loc[l_id]
        ws     = sched_cache.get((w_id, s), {})
        ls     = sched_cache.get((l_id, s), {})

        def sd(a, b, default=0.0):
            return _safe_diff(a if a is not None else default,
                              b if b is not None else default)

        base = {
            "diff_ppg":      _safe_diff(w["ppg"],        l["ppg"]),
            "diff_papg":     _safe_diff(l["papg"],       w["papg"]),
            "diff_win_pct":  _safe_diff(w["win_pct"],    l["win_pct"]),
            "diff_pt_diff":  _safe_diff(w["point_diff"], l["point_diff"]),
            "diff_recent_form":   sd(ws.get("recent_form"),   ls.get("recent_form"),   0.5),
            "diff_road_win_pct":  sd(ws.get("road_win_pct"),  ls.get("road_win_pct"),  0.5),
            "diff_margin_std":    sd(ls.get("margin_std"),    ws.get("margin_std"),    0.0),  # lower std is better, flip
            "diff_close_win_pct": sd(ws.get("close_win_pct"), ls.get("close_win_pct"), 0.5),
        }
        records.append({**base, "label": 1})
        records.append({
            "diff_ppg":           -base["diff_ppg"],
            "diff_papg":          -base["diff_papg"],
            "diff_win_pct":       -base["diff_win_pct"],
            "diff_pt_diff":       -base["diff_pt_diff"],
            "diff_recent_form":   -base["diff_recent_form"],
            "diff_road_win_pct":  -base["diff_road_win_pct"],
            "diff_margin_std":    -base["diff_margin_std"],
            "diff_close_win_pct": -base["diff_close_win_pct"],
            "label": 0,
        })

    df_out = pd.DataFrame(records).dropna()
    print(f"  {len(df_out)} training rows ({len(df_out)//2} games; {skipped} skipped).\n")
    return df_out


# ── Model training ────────────────────────────────────────────────────────────

def train_model(
    train_df: pd.DataFrame,
) -> tuple[LogisticRegression, StandardScaler]:
    """
    Fit a logistic regression on team stat diffs.

    Returns
    -------
    model  : fitted LogisticRegression
    scaler : fitted StandardScaler (must be used on inference inputs too)
    """
    X = train_df[FEATURE_COLS].values
    y = train_df["label"].values

    # Replace any inf/-inf with NaN then fill with 0 before scaling
    X = np.where(np.isfinite(X), X, 0.0)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = LogisticRegression(C=1.0, max_iter=2000, random_state=42, solver="saga", tol=1e-4)
    model.fit(X_sc, y)

    cv = cross_val_score(model, X_sc, y, cv=5, scoring="accuracy")
    print(f"  CV Accuracy : {cv.mean():.3f} ± {cv.std():.3f}")
    print(f"  Coefficients: {dict(zip(FEATURE_COLS, model.coef_[0].round(4)))}\n")
    return model, scaler


# ── Win-probability matrix ────────────────────────────────────────────────────

def build_win_prob_matrix(
    teams: pd.DataFrame,
    model: LogisticRegression,
    scaler: StandardScaler,
) -> tuple[np.ndarray, list[str]]:
    """
    Compute P[i][j] = P(team i beats team j) for every pair in `teams`.

    Returns
    -------
    P     : ndarray (N, N)
    names : list[str] aligned to matrix rows/cols
    """
    n = len(teams)
    teams = teams.reset_index(drop=True)

    feats: list[np.ndarray] = []
    idx_pairs: list[tuple[int, int]] = []

    for i in range(n):
        ti = teams.iloc[i]
        for j in range(n):
            if i == j:
                continue
            tj = teams.iloc[j]

            def _gv(series, col, default):
                v = series.get(col, default)
                try:
                    f = float(v)
                    return default if np.isnan(f) else f
                except (TypeError, ValueError):
                    return default

            feats.append(np.array([
                _safe_diff(ti["ppg"],          tj["ppg"]),
                _safe_diff(tj["papg"],         ti["papg"]),
                _safe_diff(ti["win_pct"],      tj["win_pct"]),
                _safe_diff(ti["point_diff"],   tj["point_diff"]),
                _safe_diff(_gv(ti, "recent_form",   0.5), _gv(tj, "recent_form",   0.5)),
                _safe_diff(_gv(ti, "road_win_pct",  0.5), _gv(tj, "road_win_pct",  0.5)),
                _safe_diff(_gv(tj, "margin_std",    0.0), _gv(ti, "margin_std",    0.0)),
                _safe_diff(_gv(ti, "close_win_pct", 0.5), _gv(tj, "close_win_pct", 0.5)),
            ]))
            idx_pairs.append((i, j))

    X_sc = np.clip(scaler.transform(np.array(feats)), -10, 10)
    probs = model.predict_proba(X_sc)[:, 1]

    P = np.full((n, n), 0.5)
    for (i, j), p in zip(idx_pairs, probs):
        P[i, j] = p

    return P, teams["team_name"].tolist()


# ── Display helpers ───────────────────────────────────────────────────────────

def print_top64(teams: pd.DataFrame) -> None:
    print(f"\n  {'#':<4} {'Team':<30} {'W%':>6}  {'PPG':>6}  {'PAPG':>6}  {'Diff':>6}")
    print("  " + "-" * 60)
    for rank, row in teams.iterrows():
        print(
            f"  {rank + 1:<4} {row['team_name']:<30} "
            f"{row['win_pct']:>6.3f}  {row['ppg']:>6.1f}  "
            f"{row['papg']:>6.1f}  {row['point_diff']:>+6.1f}"
        )


def print_sample(P: np.ndarray, names: list[str], n: int = 6) -> None:
    idx = list(range(min(n, len(names))))
    short = [nm[:16] for nm in names]
    header = f"{'':30s}" + "".join(f"{short[j]:>18s}" for j in idx)
    print("\nSample win-probability matrix (top-left corner):")
    print(header)
    for i in idx:
        print(f"{names[i][:29]:30s}" + "".join(f"{P[i, j]:>18.4f}" for j in idx))


def save_outputs(P: np.ndarray, names: list[str]) -> None:
    np.savez_compressed("win_prob_matrix.npz", P=P, names=np.array(names))
    print("Saved → win_prob_matrix.npz")
    pd.DataFrame(P, index=names, columns=names).to_csv(
        "win_prob_matrix.csv", float_format="%.4f"
    )
    print("Saved → win_prob_matrix.csv")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("win_probability — Win Probability Matrix")
    print("=" * 50)

    print("\n[1/4] Fetching current-season stats…")
    current = fetch_current_stats()
    if current.empty:
        print("ERROR: No stats returned.")
        sys.exit(1)

    print("\n[2/4] Fetching historical tournament games…")
    hist_games = fetch_historical_games()
    if hist_games.empty:
        print("ERROR: No historical games.")
        sys.exit(1)

    print("\n[3/4] Training logistic regression…")
    train_df = build_training_features(hist_games)
    if len(train_df) < 50:
        print("WARNING: Very few training samples.")
    model, scaler = train_model(train_df)

    print("\n[4/4] Building 64×64 win-probability matrix…")
    top64 = current.nlargest(64, "win_pct").reset_index(drop=True)
    print_top64(top64)

    P, names = build_win_prob_matrix(top64, model, scaler)
    print_sample(P, names)
    save_outputs(P, names)

    print("\nDone.")
