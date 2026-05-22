"""
Multi-agent system for the Idea Tournament.
Each agent has a specific role in generating, reviewing, evolving, and ranking hypotheses.
"""
import random, logging
from django.utils import timezone

logger = logging.getLogger(__name__)


class GenerationAgent:
    """Generates initial research hypotheses based on a research domain and question."""

    def generate(self, tournament, num_hypotheses: int = 10):
        from .models import Hypothesis, LiteratureCitation

        domains = {
            'nlp': [
                "Adaptive sparse attention for long‑context transformers",
                "Cross‑lingual knowledge distillation without parallel corpora",
                "Federated fine‑tuning of LLMs with differential privacy guarantees",
                "Quantum‑inspired tensor decomposition for model compression",
                "Neuro‑symbolic reasoning for scientific claim verification",
                "Multi‑modal alignment via contrastive learning in latent space",
                "Curriculum learning strategies for emergent abilities in LLMs",
                "Retrieval‑augmented generation with dynamic memory consolidation",
                "Adversarial robustness through architectural inductive biases",
                "Green AI: carbon‑aware training schedules for large models",
            ],
            'cv': [
                "Self‑supervised feature learning from event cameras",
                "Neural radiance fields with temporal consistency",
                "Open‑vocabulary instance segmentation via vision‑language models",
                "Geometric deep learning for protein structure prediction",
                "Efficient video transformers with token merging",
            ],
            'general': [
                "Meta‑learning for rapid domain adaptation in scientific ML",
                "Explainable AI through causal intervention analysis",
                "Automated theorem proving with large language models",
                "Reinforcement learning from human feedback for scientific discovery",
                "Multi‑agent collaboration for complex reasoning tasks",
            ],
        }

        # Pick appropriate domain templates
        domain_key = 'nlp' if 'nlp' in tournament.research_domain.lower() else \
                     'cv' if 'cv' in tournament.research_domain.lower() or 'vision' in tournament.research_domain.lower() else \
                     'general'

        templates = domains.get(domain_key, domains['general'])
        hypotheses = []

        for i in range(min(num_hypotheses, len(templates))):
            hypothesis = Hypothesis.objects.create(
                tournament=tournament,
                title=templates[i],
                content=f"Research question: {tournament.research_question}\n\n"
                        f"Proposed hypothesis: {templates[i]}\n\n"
                        f"This hypothesis explores a novel direction in {tournament.research_domain}. "
                        f"It addresses a gap in the current literature and has the potential for "
                        f"significant impact if validated.",
                round_created=1,
                status='proposed',
                generated_by='generation',
            )

            # Add a sample citation
            LiteratureCitation.objects.create(
                hypothesis=hypothesis,
                title=f"Related work in {tournament.research_domain}",
                authors=["Smith, J.", "Chen, L."],
                year=2025,
                snippet="This paper demonstrates the foundational principles that motivate this hypothesis.",
                relevance_score=0.85,
            )

            hypotheses.append(hypothesis)

        tournament.total_hypotheses_generated += len(hypotheses)
        tournament.current_round = 1
        tournament.save(update_fields=['total_hypotheses_generated', 'current_round'])

        return hypotheses


class ProximityAgent:
    """Evaluates hypotheses based on proximity to existing literature."""

    def review(self, hypothesis, round_number: int):
        from .models import HypothesisReview

        # Simulated review scores
        proximity_score = round(random.uniform(5.0, 9.0), 1)

        review = HypothesisReview.objects.create(
            hypothesis=hypothesis,
            agent='proximity',
            round_number=round_number,
            score=proximity_score,
            strengths="Well‑grounded in existing work; clearly extends current state‑of‑the‑art.",
            weaknesses="May overlap with recently published work; need to differentiate further.",
            suggestions="Consider narrowing the scope to a specific sub‑domain for greater impact.",
            overall_assessment="Solid hypothesis with clear connections to the literature.",
        )
        return review


class ReflectionAgent:
    """Critically reflects on hypotheses, identifying logical gaps and assumptions."""

    def review(self, hypothesis, round_number: int):
        from .models import HypothesisReview

        reflection_score = round(random.uniform(4.0, 8.5), 1)

        review = HypothesisReview.objects.create(
            hypothesis=hypothesis,
            agent='reflection',
            round_number=round_number,
            score=reflection_score,
            strengths="Novel angle; challenges existing assumptions in the field.",
            weaknesses="Some underlying assumptions are not fully justified; experimental design unclear.",
            suggestions="Explicitly state and justify key assumptions; propose a concrete validation method.",
            overall_assessment="Interesting but needs refinement before it can be actionable.",
        )
        return review


class RankingAgent:
    """Ranks hypotheses based on aggregated scores."""

    def rank(self, tournament, round_number: int):
        from .models import Hypothesis

        hypotheses = tournament.hypotheses.filter(round_created=round_number)
        for hyp in hypotheses:
            avg = hyp.reviews.filter(round_number=round_number).aggregate(
                avg_score=models.Avg('score')
            )['avg_score'] or 0
            hyp.score = round(avg, 2)
            hyp.save(update_fields=['score'])

        # Assign ranks
        ranked = tournament.hypotheses.filter(round_created=round_number).order_by('-score')
        for i, hyp in enumerate(ranked, 1):
            hyp.rank = i
            if i <= tournament.top_k:
                hyp.status = 'shortlisted'
            else:
                hyp.status = 'eliminated'
            hyp.save(update_fields=['rank', 'status'])

        return ranked


class EvolutionAgent:
    """Evolves the top hypotheses by combining and refining them."""

    def evolve(self, tournament, round_number: int):
        from .models import Hypothesis, LiteratureCitation

        # Get top hypotheses from previous round
        top = tournament.hypotheses.filter(
            round_created=round_number - 1, status='shortlisted'
        ).order_by('rank')[:tournament.top_k]

        new_hypotheses = []
        for i, parent in enumerate(top):
            evolved = Hypothesis.objects.create(
                tournament=tournament,
                title=f"Evolved: {parent.title}",
                content=f"Original hypothesis: {parent.content}\n\n"
                        f"Evolution: This refined version incorporates feedback from round {round_number - 1} "
                        f"and combines insights from multiple reviewers.",
                round_created=round_number,
                status='proposed',
                generated_by='evolution',
                parent_hypothesis=parent,
            )

            # Copy citations from parent
            for citation in parent.citations.all():
                LiteratureCitation.objects.create(
                    hypothesis=evolved,
                    title=citation.title,
                    authors=citation.authors,
                    year=citation.year,
                    url=citation.url,
                    snippet=citation.snippet,
                    relevance_score=citation.relevance_score,
                )

            new_hypotheses.append(evolved)

        tournament.total_hypotheses_generated += len(new_hypotheses)
        tournament.current_round = round_number
        tournament.save(update_fields=['total_hypotheses_generated', 'current_round'])

        return new_hypotheses


class MetaReviewAgent:
    """Provides a meta‑review synthesizing all reviews for a hypothesis."""

    def review(self, hypothesis, round_number: int):
        from .models import HypothesisReview

        meta_score = round(random.uniform(6.0, 9.5), 1)

        review = HypothesisReview.objects.create(
            hypothesis=hypothesis,
            agent='meta_review',
            round_number=round_number,
            score=meta_score,
            strengths="Strong consensus among reviewers on the novelty and relevance.",
            weaknesses="Minor concerns about feasibility that can be addressed in future iterations.",
            suggestions="Proceed to experimental design phase with the refined hypothesis.",
            overall_assessment="This hypothesis is ready for the next stage of research.",
        )
        return review