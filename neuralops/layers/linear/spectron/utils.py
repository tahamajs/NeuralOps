import numpy as np

def spectral_entropy(singular_values: np.ndarray) -> float:
    s2 = singular_values.astype(np.float64) ** 2
    total = s2.sum()
    if total == 0:
        return 0.0
    p = s2 / total
    p = p[p > 1e-5]
    return float(-np.sum(p * np.log(p)))

def cumulative_entropy_fraction(s: np.ndarray, max_rank: int) -> np.ndarray:
    s2 = s.astype(np.float64) ** 2
    total = s2.sum()
    if total == 0:
        return np.ones(max_rank)
    cum = np.cumsum(s2[:max_rank]) / total
    return cum

def allocate_ranks_spectron(
    weight_matrices: list[np.ndarray],
    target_budget: int,
    names: list[str],
    max_rank_factor: float = 0.75,
    lambda_range: tuple[float, float] = (1e-5, 5.0),
    steps: int = 100,
    whitening_factors: list[np.ndarray | None] | None = None,
) -> dict[str, int]:
    """
    Lagrangian optimization allocating rank per weight matrix under a
    parameter budget.
    """
    cost_tables, param_tables = [], []
    for i, W in enumerate(weight_matrices):
        d_out, d_in = W.shape
        max_r = min(d_in, d_out, max(1, int(max_rank_factor * min(d_in, d_out))))

        W_eff = W
        if whitening_factors is not None and whitening_factors[i] is not None:
            L = whitening_factors[i]
            if L.shape == (d_in, d_in):
                W_eff = W @ L

        s = np.linalg.svd(W_eff.astype(np.float64), compute_uv=False)
        cum_ent = cumulative_entropy_fraction(s, max_r)
        cost = 1.0 - cum_ent
        cost_tables.append(cost)
        param_tables.append((np.arange(1, max_r + 1) * (d_in + d_out)).astype(np.float64))

    low, high = lambda_range
    best_ranks = None
    for _ in range(steps):
        lam = 0.5 * (low + high)
        ranks = []
        total_params = 0.0
        for i in range(len(weight_matrices)):
            obj = cost_tables[i] + lam * param_tables[i]
            idx = int(np.argmin(obj))
            ranks.append(idx + 1)
            total_params += param_tables[i][idx]
        if total_params <= target_budget:
            best_ranks = ranks.copy()
            high = lam
        else:
            low = lam

    if best_ranks is None:
        best_ranks = [1] * len(weight_matrices)
    return {n: r for n, r in zip(names, best_ranks)}
