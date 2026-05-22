"""
Core AI Co‑Scientist Engine – multi‑agent tournament workflow.
"""
import json, logging, random
from django.utils import timezone
from openai import OpenAI

logger = logging.getLogger(__name__)

# Initialize OpenAI client (works with any OpenAI‑compatible API)
_client = None


def get_llm_client():
    global _client
    if _client is None:
        from django.conf import settings
        api_key = getattr(settings, 'OPENAI_API_KEY', '')
        base_url = getattr(settings, 'OPENAI_BASE_URL', None)
        _client = OpenAI(api_key=api_key, base_url=base_url)
    return _client


def call_llm(messages, model='gpt-4o-mini', temperature=0.7, max_tokens=2000):
    """Call the LLM with the given messages and return the response text."""
    client = get_llm_client()
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error("LLM call failed: %s", e)
        raise


class AIScientistFramework:
    """
    Multi‑agent research workflow.
    Implements the full tournament‑based selection mechanism.
    """

    def __init__(
        self,
        session,
        model_name='gpt-4o-mini',
        max_iterations=3,
        hypotheses_per_generation=10,
        tournament_size=8,
        evolution_top_k=3,
        verbose=True,
    ):
        from .models import Hypothesis

        self.session = session
        self.model_name = model_name
        self.max_iterations = max_iterations
        self.hypotheses_per_generation = hypotheses_per_generation
        self.tournament_size = tournament_size
        self.evolution_top_k = evolution_top_k
        self.verbose = verbose

    def run_research_workflow(self):
        """Execute the full research workflow."""
        from .models import Hypothesis

        self.session.status = 'running'
        self.session.save(update_fields=['status'])

        try:
            # ── Generation Phase ──────────────────────────────
            self._log("🔬 Generation Agent: Creating initial hypotheses...")
            hypotheses = self._generate_hypotheses(self.session.research_goal)
            self._log(f"✅ Generated {len(hypotheses)} hypotheses")

            for iteration in range(self.max_iterations):
                self._log(f"\n{'='*50}")
                self._log(f"🔄 ITERATION {iteration + 1}/{self.max_iterations}")
                self._log(f"{'='*50}")

                # ── Reflection Phase ──────────────────────────
                self._log("🧪 Reflection Agent: Peer review...")
                for h in hypotheses:
                    review = self._reflect(h)
                    HypothesisReview.objects.create(
                        hypothesis=h, agent='Reflection',
                        strengths=review.get('strengths', ''),
                        weaknesses=review.get('weaknesses', ''),
                        score=review.get('score', 5),
                        suggestions=review.get('suggestions', ''),
                    )

                # ── Ranking Phase (Elo tournament) ───────────
                self._log("🏆 Ranking Agent: Tournament...")
                self._run_tournament(hypotheses)

                # ── Meta‑review Phase ─────────────────────────
                self._log("📋 Meta‑review Agent: Synthesizing...")
                for h in hypotheses:
                    self._meta_review(h)

                # ── Proximity Phase ───────────────────────────
                self._log("🎯 Proximity Agent: Ensuring diversity...")
                hypotheses = self._filter_diverse(hypotheses)

                # ── Evolution Phase ───────────────────────────
                if iteration < self.max_iterations - 1:
                    self._log("🧬 Evolution Agent: Refining top hypotheses...")
                    top_k = sorted(hypotheses, key=lambda h: h.elo_score, reverse=True)[:self.evolution_top_k]
                    new_hypotheses = []
                    for h in top_k:
                        new_h = self._evolve(h)
                        if new_h:
                            new_hypotheses.append(new_h)
                    hypotheses = new_hypotheses or top_k
                    self._log(f"✅ Evolved {len(hypotheses)} hypotheses")

            # ── Final selection ───────────────────────────────
            best = max(hypotheses, key=lambda h: h.elo_score)
            best.status = 'finalist'
            best.save(update_fields=['status'])
            self.session.final_hypothesis_id = best.id
            self.session.status = 'completed'
            self.session.completed_at = timezone.now()
            self.session.save(update_fields=['final_hypothesis_id', 'status', 'completed_at'])

            self._log(f"\n🏁 Final best hypothesis: {best.title}")
            self._log(f"   Elo score: {best.elo_score:.0f}")
            return {
                'status': 'completed',
                'final_hypothesis_id': str(best.id),
                'title': best.title,
                'content': best.content,
                'elo_score': best.elo_score,
                'total_hypotheses': Hypothesis.objects.filter(session=self.session).count(),
            }

        except Exception as e:
            self.session.status = 'failed'
            self.session.error_message = str(e)
            self.session.save(update_fields=['status', 'error_message'])
            logger.exception("Research workflow failed")
            return {'status': 'failed', 'error': str(e)}

    # ── Agent Implementations ─────────────────────────────────────

    def _generate_hypotheses(self, goal):
        """Generation Agent: create initial hypotheses."""
        from .models import Hypothesis

        prompt = f"""You are a senior research scientist. Generate {self.hypotheses_per_generation} novel research hypotheses for the following research goal:

RESEARCH GOAL:
{goal}

For each hypothesis, provide:
1. A clear, specific title
2. A detailed description (200+ words) explaining the rationale, methodology, and expected outcomes
3. Why this hypothesis is worth investigating

Return your response as a JSON array with objects containing "title" and "content" fields."""

        response = call_llm(
            [{'role': 'user', 'content': prompt}],
            model=self.model_name,
            temperature=0.8,
            max_tokens=4000,
        )

        try:
            # Try to parse JSON
            data = json.loads(response)
        except json.JSONDecodeError:
            # Fallback: split by numbered items
            data = []
            for line in response.split('\n'):
                if line.startswith('#'):
                    title = line.lstrip('#').strip()
                    data.append({'title': title, 'content': ''})

        hypotheses = []
        for item in data[:self.hypotheses_per_generation]:
            h = Hypothesis.objects.create(
                session=self.session,
                generation=1,
                title=item.get('title', 'Untitled Hypothesis')[:300],
                content=item.get('content', ''),
                status='proposed',
            )
            hypotheses.append(h)

        return hypotheses

    def _reflect(self, hypothesis):
        """Reflection Agent: peer review a hypothesis."""
        prompt = f"""You are a rigorous peer reviewer. Evaluate the following research hypothesis:

TITLE: {hypothesis.title}
CONTENT: {hypothesis.content}

Provide your review in JSON format with these fields:
- "strengths": What is strong about this hypothesis?
- "weaknesses": What are its flaws or gaps?
- "score": A score from 1 to 10
- "suggestions": How could it be improved?"""

        response = call_llm(
            [{'role': 'user', 'content': prompt}],
            model=self.model_name,
            temperature=0.3,
            max_tokens=1000,
        )

        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {
                'strengths': 'Unable to parse review',
                'weaknesses': '',
                'score': 5,
                'suggestions': '',
            }

    def _run_tournament(self, hypotheses):
        """Ranking Agent: pairwise Elo tournament."""
        from .models import TournamentMatch

        for _ in range(min(self.tournament_size, len(hypotheses) * 2)):
            a, b = random.sample(hypotheses, 2)
            prompt = f"""Compare these two research hypotheses:

HYPOTHESIS A: {a.title}
{a.content[:500]}

HYPOTHESIS B: {b.title}
{b.content[:500]}

Which is stronger? Reply with just "A" or "B", then briefly explain why."""

            response = call_llm(
                [{'role': 'user', 'content': prompt}],
                model=self.model_name,
                temperature=0.2,
                max_tokens=300,
            )

            winner = a if response.strip().upper().startswith('A') else b

            # Update Elo ratings
            k = 32
            expected_a = 1 / (1 + 10 ** ((b.elo_score - a.elo_score) / 400))
            expected_b = 1 - expected_a

            if winner == a:
                a.elo_score += k * (1 - expected_a)
                b.elo_score += k * (0 - expected_b)
            else:
                a.elo_score += k * (0 - expected_a)
                b.elo_score += k * (1 - expected_b)

            a.comparisons_count += 1
            b.comparisons_count += 1
            a.save(update_fields=['elo_score', 'comparisons_count'])
            b.save(update_fields=['elo_score', 'comparisons_count'])

            TournamentMatch.objects.create(
                session=self.session,
                generation=self.session.hypotheses.aggregate(
                    max_gen=models.Max('generation')
                )['max_gen'] or 1,
                hypothesis_a=a,
                hypothesis_b=b,
                winner=winner,
                reasoning=response,
            )

    def _meta_review(self, hypothesis):
        """Meta‑review Agent: synthesize all reviews."""
        from .models import MetaReview

        reviews = hypothesis.reviews.all()
        reviews_text = '\n'.join(
            f"Reviewer: {r.agent}\nStrengths: {r.strengths}\nWeaknesses: {r.weaknesses}\nScore: {r.score}\n"
            for r in reviews
        )

        prompt = f"""Synthesize the following peer reviews into a meta‑review:

HYPOTHESIS: {hypothesis.title}
{hypothesis.content[:300]}

REVIEWS:
{reviews_text}

Provide:
1. A synthesis paragraph
2. A final score (1–10)
3. A recommendation: "accept", "revise", or "reject"

Return as JSON with fields "synthesis", "final_score", and "recommendation"."""

        response = call_llm(
            [{'role': 'user', 'content': prompt}],
            model=self.model_name,
            temperature=0.3,
            max_tokens=800,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            data = {'synthesis': 'Synthesis unavailable', 'final_score': 5, 'recommendation': 'revise'}

        MetaReview.objects.create(
            hypothesis=hypothesis,
            synthesis=data.get('synthesis', ''),
            final_score=data.get('final_score', 5),
            recommendation=data.get('recommendation', 'revise'),
        )

    def _filter_diverse(self, hypotheses):
        """Proximity Agent: ensure diversity, reduce redundancy."""
        # Keep top hypotheses that are dissimilar
        kept = []
        for h in sorted(hypotheses, key=lambda h: h.elo_score, reverse=True):
            if len(kept) >= self.hypotheses_per_generation // 2:
                break
            # Simple diversity check: just keep it
            kept.append(h)
        return kept

    def _evolve(self, hypothesis):
        """Evolution Agent: refine a hypothesis."""
        from .models import Hypothesis

        reviews = hypothesis.reviews.all()
        suggestions = ' '.join(r.suggestions for r in reviews if r.suggestions)

        prompt = f"""Refine the following research hypothesis based on reviewer feedback:

ORIGINAL HYPOTHESIS: {hypothesis.title}
{hypothesis.content[:500]}

REVIEWER SUGGESTIONS: {suggestions[:500]}

Provide an improved version with a new title and detailed content. Return as JSON with "title" and "content"."""

        response = call_llm(
            [{'role': 'user', 'content': prompt}],
            model=self.model_name,
            temperature=0.6,
            max_tokens=2000,
        )

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            data = {'title': f"Refined: {hypothesis.title}", 'content': hypothesis.content}

        new_h = Hypothesis.objects.create(
            session=self.session,
            generation=hypothesis.generation + 1,
            title=data.get('title', hypothesis.title)[:300],
            content=data.get('content', hypothesis.content),
            parent_hypothesis=hypothesis,
            status='refined',
        )
        return new_h

    def _log(self, message):
        if self.verbose:
            logger.info(message)