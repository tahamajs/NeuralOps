from django.urls import path

from . import views

urlpatterns = [
    # ── Reputation Scores ────────────────────────────────────
    path("scores/<str:username>/", views.score_detail, name="score_detail"),
    path("leaderboard/", views.reputation_leaderboard, name="reputation_leaderboard"),
    # ── Staking ──────────────────────────────────────────────
    path("stakes/", views.my_stakes, name="my_stakes"),
    path("stakes/stake/", views.stake_tokens, name="stake_tokens"),
    path(
        "stakes/<uuid:stake_id>/unbond/", views.start_unbonding, name="start_unbonding"
    ),
    path(
        "stakes/<uuid:stake_id>/withdraw/", views.withdraw_stake, name="withdraw_stake"
    ),
    # ── Slashing ─────────────────────────────────────────────
    path("slashing/history/", views.slashing_history, name="slashing_history"),
    path(
        "slashing/conditions/",
        views.slashing_condition_list,
        name="slashing_condition_list",
    ),
    # ── Disputes ─────────────────────────────────────────────
    path("disputes/", views.my_disputes, name="my_disputes"),
    path("disputes/raise/<uuid:job_id>/", views.raise_dispute, name="raise_dispute"),
    path("disputes/<uuid:dispute_id>/", views.dispute_detail, name="dispute_detail"),
    path("disputes/<uuid:dispute_id>/jury/", views.select_jury, name="select_jury"),
    path(
        "disputes/<uuid:dispute_id>/vote/", views.cast_jury_vote, name="cast_jury_vote"
    ),
    # ── Badges ───────────────────────────────────────────────
    path("badges/", views.badge_list, name="badge_list"),
    path("badges/mine/", views.my_badges, name="my_badges"),
    # ── Trust Graph ──────────────────────────────────────────
    path("trust/compute/", views.compute_trust_graph, name="compute_trust_graph"),
    path("trust/ranking/", views.trust_ranking, name="trust_ranking"),
    # ── Dashboard ────────────────────────────────────────────
    path("dashboard/", views.my_reputation_dashboard, name="my_reputation_dashboard"),
]
