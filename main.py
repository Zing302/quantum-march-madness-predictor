"""
main.py — March Madness Quantum-Classical Predictor
=====================================================
Orchestrates the full pipeline:

  Phase 1 (Classical):  fetch ESPN stats → train logistic model → win-prob matrix
  Phase 2 (Quantum):    encode matchup probs as Ry rotation angles → sample R1 winners
  Phase 3 (QAOA):       find globally optimal bracket via QUBO + QAOA
  Phase 4 (Eval):       backtest against historical tournament results

Cache behaviour (cache/ directory):
  --refresh   force re-fetch all data and retrain
  (default)   load from cache if available, skip API calls
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

# Phase 1 — Classical
from data.fetch_stats import fetch_current_stats, fetch_historical_games
from data.bracket_2026 import get_round1_matchups, get_round1_matchups_indexed, ESPN_NAME_MAP, BRACKET
from classical.win_probability import build_training_features, train_model, build_win_prob_matrix

# Phase 2 — Quantum circuit encoding
from quantum.matchup_circuit import build_round_circuits, sample_winners

# Phase 3 — QAOA
from quantum.bracket_qaoa import build_qubo
from quantum.simulate import run_qaoa

# Phase 4 — Eval (stub, filled in later)
# from eval.backtest import score_bracket

CACHE_DIR = Path("cache")

SEED_FALLBACK_STATS = {
    13: (0.58, 72.0, 71.0,  1.5),
    14: (0.57, 71.0, 71.5,  1.0),
    15: (0.55, 70.0, 72.0,  0.5),
    16: (0.52, 68.0, 73.0,  0.0),
}


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save(obj, path: Path):
    CACHE_DIR.mkdir(exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _seed_map() -> dict[str, int]:
    seed_order = [1, 8, 5, 4, 6, 3, 7, 2]
    opp_order  = [16, 9, 12, 13, 11, 14, 10, 15]
    smap = {}
    for region_games in BRACKET.values():
        for (a, b), s1, s2 in zip(region_games, seed_order, opp_order):
            smap[a] = s1
            smap[b] = s2
    return smap


def main(refresh: bool = False):
    # ── Phase 1: Classical win-probability matrix ─────────────────────────────
    print("=== Phase 1: Classical Model ===")

    current_cache = CACHE_DIR / "current_stats.pkl"
    if not refresh and current_cache.exists():
        print("  Loading current stats from cache…")
        current = _load(current_cache)
    else:
        current = fetch_current_stats()
        _save(current, current_cache)

    games_cache = CACHE_DIR / "hist_games.pkl"
    if not refresh and games_cache.exists():
        print("  Loading historical games from cache…")
        hist_games = _load(games_cache)
    else:
        hist_games = fetch_historical_games()
        _save(hist_games, games_cache)

    model_cache = CACHE_DIR / "model.pkl"
    if not refresh and model_cache.exists():
        print("  Loading trained model from cache…")
        model, scaler = _load(model_cache)
    else:
        train_df = build_training_features(hist_games)
        model, scaler = train_model(train_df)
        _save((model, scaler), model_cache)

    # Build tournament field DataFrame
    current["team_name"] = current["team_name"].replace(ESPN_NAME_MAP)
    tournament_teams = [t for pair in get_round1_matchups() for t in pair]
    field = current[current["team_name"].isin(tournament_teams)].drop_duplicates("team_name").reset_index(drop=True)

    # Fallback for teams missing from ESPN stats
    smap    = _seed_map()
    missing = [t for t in tournament_teams if t not in field["team_name"].values]
    if missing:
        numeric_cols = field.select_dtypes(include="number").columns
        median_stats = field[numeric_cols].median()
        fallback_rows = []
        for team in missing:
            seed = smap.get(team, 14)
            wp, ppg, papg, pd_ = SEED_FALLBACK_STATS.get(seed, (0.55, 70.0, 72.0, 0.5))
            row = median_stats.to_dict()
            row.update({"team_name": team, "win_pct": wp, "ppg": ppg, "papg": papg, "point_diff": pd_})
            fallback_rows.append(row)
            print(f"  Fallback stats inserted for '{team}' (seed {seed})")
        field = pd.concat([field, pd.DataFrame(fallback_rows)], ignore_index=True)

    sched_cache_path = CACHE_DIR / "schedule_features.pkl"
    if not refresh and sched_cache_path.exists():
        print("  Loading schedule features from cache…")
        sched_feats = _load(sched_cache_path)
    else:
        print("  Fetching regular season schedule features for tournament teams…")
        from data.fetch_stats import fetch_team_schedule_features
        sched_feats = {}
        team_ids = {}
        if "team_id" in field.columns:
            for _, row in field.iterrows():
                tid = row.get("team_id")
                if pd.notna(tid) and str(tid) not in ("nan", ""):
                    team_ids[row["team_name"]] = str(tid)
        for tname, tid in team_ids.items():
            feats = fetch_team_schedule_features(str(tid))
            if feats:
                sched_feats[tname] = feats
            time.sleep(0.2)
        _save(sched_feats, sched_cache_path)

    for col in ["recent_form", "road_win_pct", "margin_std", "close_win_pct"]:
        field[col] = field["team_name"].map(
            lambda t: sched_feats.get(t, {}).get(col, None)
        )
    print(f"  Schedule features loaded for {sum(1 for v in sched_feats.values() if v)} teams")

    P, names = build_win_prob_matrix(field, model, scaler)
    matchups  = get_round1_matchups_indexed(names)

    print(f"  Win-probability matrix: {P.shape}  |  matchups resolved: {len(matchups)}/32")

    # ── Phase 2: Quantum circuit encoding ─────────────────────────────────────
    print("\n=== Phase 2: Quantum Circuit Encoding ===")
    circuits = build_round_circuits(P, matchups)
    print(f"  Built {len(circuits)} parameterized circuits")

    r1_winners = sample_winners(circuits, matchups, shots=1024)
    print(f"  Round 1 winners (team indices): {r1_winners}")
    print(f"  Round 1 winners (team names):   {[names[i] for i in r1_winners]}")

    # ── Phase 3: QAOA bracket optimization ───────────────────────────────────
    print("\n=== Phase 3: QAOA Bracket Optimizer ===")
    qubo   = build_qubo(P, names, matchups)
    run_qaoa(qubo, reps=1)

    # ── Simulate full bracket using actual matchup structure ──────────────────
    # R1: quantum-sampled winners; R2+: P matrix picks winner of each actual game
    name_to_idx = {n: i for i, n in enumerate(names)}

    def _pick(a: str, b: str) -> str:
        return a if P[name_to_idx[a], name_to_idx[b]] >= 0.5 else b

    # R1 winners per actual pair (from Phase 2 quantum sampling)
    r1_winner_map: dict[str, str] = {}
    for (ia, ib), w in zip(matchups, r1_winners):
        winner = names[w]
        r1_winner_map[names[ia]] = winner
        r1_winner_map[names[ib]] = winner

    def _game(a: str, b: str, use_p: bool = True) -> dict:
        winner = _pick(a, b) if use_p else r1_winner_map.get(a, r1_winner_map.get(b))
        return {"teams": [a, b], "winner": winner}

    regions_data = {}
    for region, region_matchups in BRACKET.items():
        # R1 — quantum sampling
        r1 = [_game(a, b, use_p=False) for a, b in region_matchups]
        r1w = [g["winner"] for g in r1]

        # R2 — pair adjacent R1 winners: (0,1), (2,3), (4,5), (6,7)
        r2 = [_game(r1w[i], r1w[i + 1]) for i in range(0, 8, 2)]
        r2w = [g["winner"] for g in r2]

        # S16 — pair adjacent R2 winners: (0,1), (2,3)
        s16 = [_game(r2w[i], r2w[i + 1]) for i in range(0, 4, 2)]
        s16w = [g["winner"] for g in s16]

        # E8 — single regional final
        e8 = _game(s16w[0], s16w[1])

        regions_data[region] = {"r1": r1, "r2": r2, "s16": s16, "e8": e8}

    # Final Four: East vs West, South vs Midwest
    ew = regions_data["East"]["e8"]["winner"]
    ww = regions_data["West"]["e8"]["winner"]
    sw = regions_data["South"]["e8"]["winner"]
    mw = regions_data["Midwest"]["e8"]["winner"]

    f4 = [_game(ew, ww), _game(sw, mw)]
    champ = _game(f4[0]["winner"], f4[1]["winner"])

    bracket_json = {
        "generated_at": datetime.now().isoformat(),
        "regions":      regions_data,
        "final_four":   f4,
        "championship": champ,
        "champion":     champ["winner"],
    }
    Path("bracket.json").write_text(json.dumps(bracket_json, indent=2))

    print("\n  Predicted bracket:")
    for region, rd in regions_data.items():
        print(f"    {region}: R1→ {[g['winner'] for g in rd['r1']]}")
    for m in f4:
        print(f"    F4: {m['teams'][0]} vs {m['teams'][1]} → {m['winner']}")
    print(f"    Championship: {champ['teams'][0]} vs {champ['teams'][1]} → {champ['winner']}")
    print(f"    Champion: {champ['winner']}")
    print("\n  Bracket saved → bracket.json  (run 'python app.py' to view)")

    # ── Phase 4: Backtest ─────────────────────────────────────────────────────
    # print("\n=== Phase 4: Evaluation ===")
    # score = score_bracket(bracket_indices, actual_results)
    # print(f"  Bracket score: {score}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Re-fetch all data and retrain")
    args = parser.parse_args()
    main(refresh=args.refresh)
