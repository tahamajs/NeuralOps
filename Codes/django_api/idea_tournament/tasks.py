"""Celery tasks for running the idea tournament."""

import logging

from celery import shared_task

from .models import IdeaTournament, TournamentLog

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2)
def run_tournament(self, tournament_id: str):
    """Execute the full tournament: generate, review, rank, evolve, repeat."""
    try:
        tournament = IdeaTournament.objects.get(id=tournament_id)
    except IdeaTournament.DoesNotExist:
        return

    if tournament.status != "running":
        return

    from .agents import (
        EvolutionAgent,
        GenerationAgent,
        MetaReviewAgent,
        ProximityAgent,
        RankingAgent,
        ReflectionAgent,
    )

    gen_agent = GenerationAgent()
    prox_agent = ProximityAgent()
    refl_agent = ReflectionAgent()
    rank_agent = RankingAgent()
    evol_agent = EvolutionAgent()
    meta_agent = MetaReviewAgent()

    # ── Round 1: Generate initial hypotheses ──────────────────
    logger.info("Tournament %s: Generating initial hypotheses", tournament_id)
    gen_agent.generate(tournament, tournament.num_hypotheses)

    TournamentLog.objects.create(
        tournament=tournament,
        round_number=1,
        agent="generation",
        action=f"Generated {tournament.num_hypotheses} initial hypotheses",
    )

    # ── Review Round 1 ────────────────────────────────────────
    for hyp in tournament.hypotheses.filter(round_created=1):
        prox_agent.review(hyp, 1)
        refl_agent.review(hyp, 1)
        meta_agent.review(hyp, 1)

    TournamentLog.objects.create(
        tournament=tournament,
        round_number=1,
        agent="proximity,reflection,meta_review",
        action="Completed first round reviews",
    )

    # ── Rank Round 1 ──────────────────────────────────────────
    rank_agent.rank(tournament, 1)

    TournamentLog.objects.create(
        tournament=tournament,
        round_number=1,
        agent="ranking",
        action=f"Top {tournament.top_k} hypotheses shortlisted",
    )

    # ── Rounds 2..N: Evolve, Review, Rank ─────────────────────
    for round_num in range(2, tournament.num_rounds + 1):
        evol_agent.evolve(tournament, round_num)

        TournamentLog.objects.create(
            tournament=tournament,
            round_number=round_num,
            agent="evolution",
            action=f"Evolved top hypotheses for round {round_num}",
        )

        for hyp in tournament.hypotheses.filter(round_created=round_num):
            prox_agent.review(hyp, round_num)
            refl_agent.review(hyp, round_num)
            meta_agent.review(hyp, round_num)

        rank_agent.rank(tournament, round_num)

        TournamentLog.objects.create(
            tournament=tournament,
            round_number=round_num,
            agent="ranking",
            action=f"Round {round_num} ranking complete",
        )

    # ── Final: Mark top hypothesis as winner ───────────────────
    winner = (
        tournament.hypotheses.filter(
            round_created=tournament.num_rounds, status="shortlisted"
        )
        .order_by("rank")
        .first()
    )

    if winner:
        winner.status = "winner"
        winner.save(update_fields=["status"])

    tournament.status = "completed"
    tournament.completed_at = timezone.now()
    tournament.save(update_fields=["status", "completed_at"])

    TournamentLog.objects.create(
        tournament=tournament,
        round_number=tournament.num_rounds,
        agent="system",
        action=f'Tournament completed. Winner: {winner.title if winner else "N/A"}',
    )

    return f"Tournament {tournament_id} completed"
