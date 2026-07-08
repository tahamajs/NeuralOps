"""
EigenTrust-style Reputation Algorithm (OpenRank-compatible).

Computes global trust scores from a directed trust graph using the
EigenTrust algorithm: a PageRank-like eigenvector computation that
converges to a global trust vector.

Key concepts:
  - Pre-trusted peers: platform-curated trusted nodes (staff, verified validators)
  - Trust matrix normalization: each row sums to 1
  - Eigenvector computation: power iteration with damping factor α
  - Convergence: stop when ||t_{k+1} - t_k|| < epsilon

Reference: Kamvar et al., "The Eigentrust Algorithm for Reputation
           Management in P2P Networks", WWW 2003.
"""

import numpy as np
from django.contrib.auth import get_user_model

from .models import EigenTrustResult, ReputationGraph, TrustEdge

User = get_user_model()


# Pre-trusted peers: staff users and validators with high accuracy
def get_pre_trusted_peers():
    """Return the set of pre-trusted peer user IDs."""
    trusted_ids = set()
    # Staff users
    trusted_ids.update(User.objects.filter(is_staff=True).values_list("id", flat=True))
    # Validators with high accuracy
    try:
        from gpu_validator.models import ValidatorNode

        trusted_ids.update(
            ValidatorNode.objects.filter(
                accuracy_score__gte=90.0, status="active"
            ).values_list("user_id", flat=True)
        )
    except ImportError:
        pass
    # Providers with high reputation
    try:
        from .models import ReputationScore

        trusted_ids.update(
            ReputationScore.objects.filter(provider_score__gte=80.0).values_list(
                "user_id", flat=True
            )
        )
    except ImportError:
        pass
    return trusted_ids


def compute_eigentrust(
    graph: ReputationGraph,
    alpha: float = 0.85,
    epsilon: float = 1e-6,
    max_iterations: int = 100,
):
    """
    Compute EigenTrust scores for a reputation graph.

    Args:
        graph: ReputationGraph instance with TrustEdges
        alpha: Damping factor (like PageRank), controls weight of pre-trusted peers
        epsilon: Convergence threshold
        max_iterations: Maximum power iterations

    Returns:
        dict: {user_id: global_trust_score}
    """
    edges = TrustEdge.objects.filter(graph=graph).select_related("source", "target")
    if not edges.exists():
        return {}

    # Build set of all users in the graph
    all_users = set()
    for e in edges:
        all_users.add(e.source_id)
        all_users.add(e.target_id)

    # Add pre-trusted peers
    pre_trusted = get_pre_trusted_peers()
    all_users.update(pre_trusted)

    # Map user IDs to matrix indices
    user_list = sorted(all_users)
    n = len(user_list)
    user_to_idx = {uid: i for i, uid in enumerate(user_list)}

    # Build trust matrix C (normalized so each column sums to 1)
    C = np.zeros((n, n))
    col_sums = np.zeros(n)
    for e in edges:
        src = user_to_idx.get(e.source_id)
        tgt = user_to_idx.get(e.target_id)
        if src is not None and tgt is not None:
            C[src, tgt] = e.trust_score
            col_sums[tgt] += e.trust_score

    # Normalize columns (if column has trust, normalize; else uniform)
    for j in range(n):
        if col_sums[j] > 0:
            C[:, j] /= col_sums[j]
        else:
            C[:, j] = 1.0 / n

    # Pre-trusted vector p
    p = np.zeros(n)
    for uid in pre_trusted:
        idx = user_to_idx.get(uid)
        if idx is not None:
            p[idx] = 1.0 / len(pre_trusted) if pre_trusted else 1.0 / n

    # If no pre-trusted peers, use uniform distribution
    if not pre_trusted:
        p = np.ones(n) / n

    # Power iteration: t = (1 - α) * C^T * t + α * p
    # Note: EigenTrust uses C^T (rows as outgoing trust, columns as incoming)
    t = np.ones(n) / n  # initial uniform trust
    CT = C.T
    for iteration in range(max_iterations):
        t_new = (1 - alpha) * CT.dot(t) + alpha * p
        diff = np.linalg.norm(t_new - t, ord=1)
        t = t_new
        if diff < epsilon:
            break

    # Store results
    results = {}
    for i, uid in enumerate(user_list):
        score = float(t[i])
        results[uid] = score

    # Save to DB
    EigenTrustResult.objects.filter(graph=graph).delete()
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for rank, (uid, score) in enumerate(sorted_results, 1):
        EigenTrustResult.objects.create(
            graph=graph,
            user_id=uid,
            global_trust_score=round(score, 6),
            rank=rank,
        )

    return results


def build_reputation_graph(name: str = "global"):
    """
    Build a reputation graph from all available trust signals:
    - Direct job interactions (provider ↔ consumer)
    - Benchmark scores
    - Jury participation
    """
    graph = ReputationGraph.objects.create(name=name)

    # Add edges from job completions
    from gpu_marketplace.models import ComputeJob

    jobs = ComputeJob.objects.filter(status="completed").select_related(
        "consumer", "provider"
    )
    for job in jobs:
        # Consumer trusts provider (if job succeeded)
        trust = 0.8 if job.verification_proof else 0.5
        TrustEdge.objects.get_or_create(
            graph=graph,
            source=job.consumer,
            target=job.provider,
            defaults={"trust_score": trust, "derivation": "direct"},
        )
        # Provider trusts consumer (if payment was smooth)
        TrustEdge.objects.get_or_create(
            graph=graph,
            source=job.provider,
            target=job.consumer,
            defaults={"trust_score": 0.7, "derivation": "direct"},
        )

    return graph
