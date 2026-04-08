"""
matchup_circuit.py — Quantum Matchup Encoding
==============================================
Encodes win probabilities from the classical model as Ry rotation angles
on 1-qubit parameterized circuits.

    Ry(θ) |0⟩  where  θ = 2·arccos(√p)
    → P(measuring |0⟩) = p   (team i wins)
    → P(measuring |1⟩) = 1-p (team j wins)

Public API
----------
build_matchup_circuit()                -> (QuantumCircuit, Parameter)
bind_circuit(qc, theta, p)             -> QuantumCircuit
build_round_circuits(P, matchups)      -> list[QuantumCircuit]
sample_winners(circuits, matchups, shots, backend) -> list[int]
seed_to_matchups(n_teams)              -> list[tuple[int, int]]
"""

from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


# ── Core encoding ─────────────────────────────────────────────────────────────

def build_matchup_circuit() -> tuple[QuantumCircuit, Parameter]:
    """
    Returns a 1-qubit parameterized circuit with a single Parameter θ.
    Bind θ = 2·arccos(√p) before running.
    """
    theta = Parameter("θ")
    qc = QuantumCircuit(1, 1)
    qc.ry(theta, 0)
    qc.measure(0, 0)
    return qc, theta


def prob_to_angle(p: float) -> float:
    """Map win probability p ∈ [0,1] → Ry rotation angle θ."""
    p = float(np.clip(p, 1e-9, 1 - 1e-9))  # avoid arccos domain errors
    return 2.0 * np.arccos(np.sqrt(p))


def bind_circuit(qc: QuantumCircuit, theta: Parameter, p: float) -> QuantumCircuit:
    """Return a new circuit with θ bound to the angle for win probability p."""
    return qc.assign_parameters({theta: prob_to_angle(p)})


# ── Round construction ────────────────────────────────────────────────────────

def build_round_circuits(
    P: np.ndarray,
    matchups: list[tuple[int, int]],
) -> list[QuantumCircuit]:
    """
    Build one bound circuit per matchup.

    Args:
        P       : N×N win-probability matrix (from classical model)
        matchups: list of (team_i_idx, team_j_idx) pairs for this round

    Returns:
        List of bound QuantumCircuits, one per matchup.
        Measurement outcome 0 → team_i wins, 1 → team_j wins.
    """
    template, theta = build_matchup_circuit()
    circuits = []
    for i, j in matchups:
        p_ij = P[i, j]
        circuits.append(bind_circuit(template, theta, p_ij))
    return circuits


# ── Sampling ──────────────────────────────────────────────────────────────────

def sample_winners(
    circuits: list[QuantumCircuit],
    matchups: list[tuple[int, int]],
    shots: int = 1024,
    backend=None,
) -> list[int]:
    """
    Run circuits and return the winning team index for each matchup.

    Args:
        circuits : bound circuits from build_round_circuits()
        matchups : same (i, j) pairs used to build circuits
        shots    : number of measurement shots per circuit
        backend  : Qiskit backend; defaults to AerSimulator

    Returns:
        List of team indices (one winner per matchup).
    """
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    backend = AerSimulator()
    compiled = transpile(circuits, backend)
    job = backend.run(compiled, shots=shots)
    counts_list = job.result().get_counts()

    if isinstance(counts_list, dict):
        counts_list = [counts_list]

    winners = []
    for (i, j), counts in zip(matchups, counts_list):
        zeros = counts.get("0", 0)
        ones  = counts.get("1", 0)
        winner = i if zeros >= ones else j
        winners.append(winner)

    return winners


# ── Round 1 seeding helper ────────────────────────────────────────────────────

def seed_to_matchups(n_teams: int = 64) -> list[tuple[int, int]]:
    """
    Standard NCAA bracket seeding: 1v16, 2v15, ... 8v9 per region.
    Assumes teams are sorted by BPI rank (index 0 = strongest).
    Returns 32 matchup pairs for Round 1.
    """
    matchups = []
    region_size = n_teams // 4  # 16 teams per region
    seed_pairs = [(0, 15), (7, 8), (4, 11), (3, 12),
                  (5, 10), (2, 13), (6, 9),  (1, 14)]
    for region in range(4):
        offset = region * region_size
        for s1, s2 in seed_pairs:
            matchups.append((offset + s1, offset + s2))
    return matchups


# ── Main (smoke test) ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    np.random.seed(42)
    N = 64
    # Fake win-prob matrix for testing (replace with classical model output)
    raw = np.random.rand(N, N)
    P = raw / (raw + raw.T)  # ensure P[i,j] + P[j,i] = 1
    np.fill_diagonal(P, 0.5)

    matchups = seed_to_matchups(N)
    print(f"Round 1 matchups: {len(matchups)}")

    circuits = build_round_circuits(P, matchups)
    print(f"Built {len(circuits)} circuits")

    winners = sample_winners(circuits, matchups, shots=1024)
    print(f"Winners (team indices): {winners}")

    upsets = sum(1 for (i, _), w in zip(matchups, winners) if w != i)
    print(f"Upsets: {upsets}/32")
