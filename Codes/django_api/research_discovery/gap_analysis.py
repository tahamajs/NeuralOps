"""AI Research Gap Analysis Engine. Based on GAPMAP framework."""
import json, logging
from django.utils import timezone
from .models import ResearchGapAnalysis

logger = logging.getLogger(__name__)


class GapAnalyzer:
    """Analyzes research literature to identify knowledge gaps."""

    @staticmethod
    def analyze(papers: list, domain: str, user_id: int, model_id: str = 'gpt-4o') -> ResearchGapAnalysis:
        """Analyze a corpus of papers for research gaps."""
        from llm_hub.gateway import LLMGateway

        analysis = ResearchGapAnalysis.objects.create(
            user_id=user_id,
            title=f"Gap Analysis: {domain}",
            domain=domain,
            papers_compared=[p.get('id', '') for p in papers],
            status='processing',
        )

        try:
            # Build paper context
            paper_texts = '\n\n'.join(
                f"Paper {i+1}: {p.get('title', '')}\nAbstract: {p.get('abstract', '')[:1000]}"
                for i, p in enumerate(papers[:15])
            )

            # Step 1: Identify explicit gaps
            explicit_prompt = f"""Identify all explicitly stated research gaps, limitations, or future work mentioned in these papers. Return a JSON list.

{paper_texts}"""
            explicit_result = LLMGateway.complete(model_id, [{'role': 'user', 'content': explicit_prompt}], temperature=0.2)
            try:
                explicit_gaps = json.loads(explicit_result.get('content', '[]'))
            except:
                explicit_gaps = []

            # Step 2: Identify implicit gaps
            implicit_prompt = f"""Based on these papers, identify implicit research gaps—areas that are understudied or not addressed at all. Return a JSON list with gap description and reasoning.

{paper_texts}"""
            implicit_result = LLMGateway.complete(model_id, [{'role': 'user', 'content': implicit_prompt}], temperature=0.4)
            try:
                implicit_gaps = json.loads(implicit_result.get('content', '[]'))
            except:
                implicit_gaps = []

            # Step 3: Methodology gaps
            methodology_prompt = f"""Identify methodological gaps across these papers—approaches that should have been used but weren't. Return JSON list.

{paper_texts}"""
            method_result = LLMGateway.complete(model_id, [{'role': 'user', 'content': methodology_prompt}], temperature=0.3)
            try:
                methodology_gaps = json.loads(method_result.get('content', '[]'))
            except:
                methodology_gaps = []

            analysis.explicit_gaps = explicit_gaps
            analysis.implicit_gaps = implicit_gaps
            analysis.methodology_gaps = methodology_gaps
            analysis.status = 'completed'
            analysis.completed_at = timezone.now()
            analysis.save()

        except Exception as e:
            logger.exception("Gap analysis failed")
            analysis.status = 'failed'
            analysis.save()

        return analysis