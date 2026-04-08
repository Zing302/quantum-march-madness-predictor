import numpy as np
from qiskit_optimization import QuadraticProgram


def potential_opp(t: int, r: int) -> range:
    """Return the indices of teams that could face team t in round r."""
    group_size  = 2 ** (r + 1)
    group_start = (t // group_size) * group_size
    group_end   = group_start + group_size
    half_size   = group_size // 2
    if t < group_start + half_size:
        return range(group_start + half_size, group_end)
    else:
        return range(group_start, group_start + half_size)


def build_qubo(P: np.ndarray, names: list[str], round1_matchups: list[tuple[int, int]]) -> QuadraticProgram:
    # names is used downstream in decode_result() to map indices → team names

    qp = QuadraticProgram()
    for t in range(64):
        for r in range(6):
            qp.binary_var(name=f"x_{t}_{r}")

    # ── Survival probabilities ────────────────────────────────────────────────
    survive = np.zeros((64, 6))
    for t, opp in round1_matchups:
        survive[t][0]   = P[t][opp]
        survive[opp][0] = P[opp][t]
    for r in range(1, 6):
        for t in range(64):
            for opp in potential_opp(t, r):
                survive[t][r] += P[t][opp] * survive[opp][r - 1]

    # ── Objective: maximize expected ESPN score ───────────────────────────────
    ESPN_points = [10, 20, 40, 80, 160, 320]
    linear = {f"x_{t}_{r}": ESPN_points[r] * survive[t][r]
              for t in range(64) for r in range(6)}
    qp.maximize(linear=linear)

    # ── Constraints ───────────────────────────────────────────────────────────
    # Exactly one winner per bracket slot (one team per group advances each round)
    # r=0: groups of 2  → 32 R1 winners
    # r=1: groups of 4  → 16 R2 winners
    # ...
    # r=5: group of 64  →  1 champion
    for r in range(6):
        group_size = 2 ** (r + 1)
        for g in range(64 // group_size):
            group_start = g * group_size
            slot = {f"x_{t}_{r}": 1 for t in range(group_start, group_start + group_size)}
            qp.linear_constraint(linear=slot, sense="==", rhs=1, name=f"one_winner_r{r}_g{g}")

    return qp
