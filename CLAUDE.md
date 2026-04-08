# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

All commands must run inside the `qiskitEnv` conda environment:

```bash
conda activate qiskitEnv
python main.py              # full pipeline (uses cache by default)
python main.py --refresh    # force re-fetch all ESPN data and retrain model
python app.py               # start bracket viewer at http://localhost:8080
```

Port 5000 is taken by macOS AirPlay — always use 8080.

Key packages: `qiskit 2.3.0`, `qiskit-aer 0.17.1`, `qiskit-optimization 0.7.0`, `qiskit-algorithms 0.4.0`, `scikit-learn`, `flask`.

## Pipeline Architecture

Four sequential phases, all orchestrated by `main.py`:

**Phase 1 — Classical** (`data/`, `classical/`)
- `data/fetch_stats.py` pulls ESPN API → team aggregate stats (PPG, PAPG, win%, point diff) + per-team regular season schedule features (recent form, road win%, margin std, close game win%)
- `classical/win_probability.py` trains a logistic regression on historical tournament games (2010–2024) using stat diffs as features, then produces a 64×64 win-probability matrix `P[i][j]`
- ESPN API endpoints: `/teams` (paginated, 4 pages), `/teams/{id}` for records, `/teams/{id}/schedule` for game logs, `/scoreboard?seasontype=3` for historical tournament results

**Phase 2 — Quantum** (`quantum/matchup_circuit.py`)
- Each R1 matchup gets a 1-qubit circuit: `Ry(2·arccos(√p))|0⟩`
- `P(measure |0⟩) = p` — win probability is encoded as quantum amplitude
- Aer simulator samples winners; this is genuine quantum computation, not `random() < p`
- Used for mid-tournament predictions too: `build_round_circuits(P, matchups)` + `sample_winners(...)`

**Phase 3 — QUBO** (`quantum/bracket_qaoa.py`, `quantum/simulate.py`)
- Formulates bracket selection as a QUBO over 384 binary variables (64 teams × 6 rounds)
- Survival probabilities `survive[t][r]` encode expected ESPN score contribution
- Solved greedily (QAOA on 384 qubits is infeasible on classical simulators; formulation is hardware-ready)
- `run_qaoa()` in `simulate.py` is the entry point; returns a `_GreedyResult` mimicking `SampleOptimizationResult`

**Phase 3b — Bracket Simulation** (in `main.py`)
- The display bracket is NOT from the QUBO — it's simulated using Phase 2 quantum sampling for R1, then `P[i][j]` picks for R2 through Championship, following the actual bracket structure
- R2 pairings: adjacent R1 winners within each region (games 0&1, 2&3, 4&5, 6&7)
- Final Four: East vs West, South vs Midwest

**Phase 4 — Eval** (`eval/backtest.py`) — stub, pending real tournament results
- `score_bracket(predicted, actual)` in `simulate.py` is already implemented (ESPN scoring: 10/20/40/80/160/320 per round)

## Caching

All expensive fetches are pickled under `cache/`:
- `current_stats.pkl` — current season ESPN stats
- `hist_games.pkl` — historical tournament results 2010–2024
- `model.pkl` — trained LogisticRegression + StandardScaler
- `schedule_features.pkl` — 2026 regular season schedule features for tournament teams

Delete `model.pkl` when `FEATURE_COLS` in `win_probability.py` changes (model/scaler must match feature count). Delete `schedule_features.pkl` to re-fetch schedule data.

## Key Design Decisions

- `data/bracket_2026.py` contains the official bracket and `ESPN_NAME_MAP` (ESPN full names → bracket short names). Add entries here when ESPN returns unrecognized team names.
- Hawaii and Queens (NC) are not in ESPN stats — fallback stats are inserted in `main.py` using `SEED_FALLBACK_STATS`.
- Schedule features use `_gv()` helper in `build_win_prob_matrix` to handle NaN gracefully (fallback teams lack schedule data).
- `qiskit.primitives.Sampler` was removed in Qiskit 2.x — use `qiskit_aer.primitives.Sampler` instead.
- Historical schedule features have near-zero model coefficients (ESPN API returns sparse data for older seasons) — the 4 new feature columns still flow through inference but have minimal effect on P.
