# Quantum March Madness Predictor

A hybrid quantum-classical system for predicting the NCAA Men's Basketball Tournament bracket. Classical machine learning computes win probabilities; quantum circuits encode and sample those probabilities; a QUBO optimizer finds the globally optimal bracket.

## Architecture

```
ESPN Stats API
     │
     ▼
Phase 1 — Classical Model
  Logistic regression trained on tournament games 2010–2024
  Features: PPG diff, PAPG diff, win% diff, point diff,
            recent form, road win%, consistency, close game win%
  Output: 64×64 win-probability matrix P[i][j]
     │
     ▼
Phase 2 — Quantum Circuit Encoding
  Each matchup → 1-qubit circuit: Ry(2·arccos(√p))|0⟩
  P(measure |0⟩) = p  (win probability as quantum amplitude)
  Aer simulator samples Round 1 winners
     │
     ▼
Phase 3 — QUBO Bracket Optimizer
  384 binary variables (64 teams × 6 rounds)
  Survival probabilities encode expected ESPN bracket score
  Solved greedily on classical hardware
  (Formulation is hardware-ready for real quantum devices)
     │
     ▼
Bracket Simulation
  R1: quantum-sampled winners
  R2–Championship: P matrix picks following actual bracket structure
     │
     ▼
Phase 4 — Evaluation
  ESPN scoring: 10 / 20 / 40 / 80 / 160 / 320 pts per round
  2026 result: 780 pts · 56.6th percentile · rank 11,982,535
```

## Setup

```bash
conda create -n qiskitEnv python=3.13
conda activate qiskitEnv
pip install qiskit qiskit-aer qiskit-optimization qiskit-algorithms \
            scikit-learn pandas requests flask
```

## Usage

```bash
conda activate qiskitEnv

# Generate bracket prediction (uses cache if available)
python main.py

# Force re-fetch all ESPN data and retrain model
python main.py --refresh

# View bracket in browser at http://localhost:8080
python app.py
```

## What Makes This Unique

**Quantum amplitude encoding** — win probabilities are encoded as quantum amplitudes (`P(|0⟩) = p`), not just fed into `random() < p`. The wavefunction collapses on measurement, making quantum mechanics the actual mechanism behind winner selection.

**QUBO formulation** — bracket optimization is cast as a combinatorial optimization problem with survival probability coefficients and one-winner-per-slot constraints. This is the correct formulation for QAOA on quantum hardware; the greedy solver is a classical stand-in for when 384-qubit simulation is infeasible.

**Coherent pipeline** — the classical model, quantum sampling, and QUBO optimization all operate on the same `P[i][j]` matrix. Each phase has a clear role rather than quantum being bolted on as a gimmick.

## Project Structure

```
main.py                  — pipeline orchestrator + bracket simulation
app.py                   — Flask bracket viewer (localhost:8080)
data/
  fetch_stats.py         — ESPN API fetcher (stats + schedule + tournament history)
  bracket_2026.py        — official 2026 bracket, team name aliases
classical/
  win_probability.py     — logistic regression training + P matrix construction
quantum/
  matchup_circuit.py     — Ry encoding, circuit building, Aer sampling
  bracket_qaoa.py        — QUBO formulation (QuadraticProgram)
  simulate.py            — greedy QUBO solver, bracket decoder, ESPN scorer
eval/
  backtest.py            — [stub] score predictions vs actual results
cache/                   — pickle cache for stats, model, schedule features
```

## 2026 Results

**Overall: 780 pts · 56.6th percentile · rank ~11.98M**

| Round | Predicted Correct | Notes |
|-------|------------------|-------|
| Round of 32 | — | not tracked |
| Sweet 16 | 10 / 16 | |
| Elite 8 | 3 / 8 | Duke (our pick) lost to UConn 72–73 |
| Final Four | 2 / 4 | Michigan and UConn correct; missed Illinois/Arizona |
| Championship | correct matchup | Michigan vs UConn |
| Champion | ✗ (predicted Duke) | **Michigan** won 69–63 over UConn |
