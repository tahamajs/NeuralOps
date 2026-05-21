"""
Kleros-style Random Jury Selection Engine.

Selects jurors from staked users with probability proportional to stake.
Inspired by Kleros: "the more PNK a user stakes, the more chance he has
of being chosen as a juror."

Uses weighted random selection without replacement.
"""
import random
import logging
from django.contrib.auth import get_user_model
from .models import Stake, Dispute

User = get_user_model()
logger = logging.getLogger(__name__)


def select_jurors(dispute: Dispute, num_jurors: int) -> list:
    """
    Select `num_jurors` from the pool of staked users.
    Selection probability is proportional to the user's stake amount.
    Excludes the disputing parties.
    """
    # Pool: active stakes, excluding the two disputing parties
    stakes = Stake.objects.filter(
        status='active',
    ).exclude(
        user__in=[dispute.raised_by, dispute.against]
    ).select_related('user')

    if not stakes.exists():
        # Fallback: include all active users (not just stakers)
        all_users = User.objects.filter(is_active=True).exclude(
            id__in=[dispute.raised_by_id, dispute.against_id]
        )
        if all_users.count() < num_jurors:
            logger.warning("Not enough users for jury selection")
            return random.sample(list(all_users), min(num_jurors, all_users.count()))
        return random.sample(list(all_users), num_jurors)

    # Weighted random selection without replacement
    users = []
    weights = []
    for stake in stakes:
        if stake.user_id not in [u.id for u in users]:
            users.append(stake.user)
            weights.append(float(stake.amount_usd))

    if len(users) < num_jurors:
        logger.warning(f"Only {len(users)} staked users available for jury of {num_jurors}")
        return users

    # Weighted random selection without replacement
    selected = []
    remaining_users = list(users)
    remaining_weights = list(weights)

    for _ in range(num_jurors):
        # Normalize weights
        total = sum(remaining_weights)
        if total == 0:
            break
        probs = [w / total for w in remaining_weights]

        # Select one
        idx = random.choices(range(len(remaining_users)), weights=probs, k=1)[0]
        selected.append(remaining_users[idx])
        # Remove selected
        remaining_users.pop(idx)
        remaining_weights.pop(idx)

    return selected