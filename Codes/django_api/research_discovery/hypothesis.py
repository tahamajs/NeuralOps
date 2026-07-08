"""
AI Hypothesis Generation Engine.
Inspired by: HypER (Literature-grounded Hypothesis Generation),
HypoAgents (Bayes-Entropy Collaborative Multi-Agent),
NSHE (Neuro-Symbolic Hypothesis Engine).
"""

import json
import logging

from .models import GeneratedHypothesis, HypothesisSource

logger = logging.getLogger(__name__)


class HypothesisGenerator:
    """Generates novel, testable hypotheses from scientific literature."""

    @staticmethod
    def generate(
        domain: str,
        papers: list,
        user_id: int,
        model_id: str = "gpt-4o",
        num_hypotheses: int = 3,
    ) -> list:
        """Generate hypotheses from a corpus of papers."""
        from llm_hub.gateway import LLMGateway

        # Step 1: Extract key claims and gaps from papers
        paper_claims = []
        for paper in papers[:20]:
            claims = HypothesisGenerator._extract_claims(paper, model_id)
            paper_claims.append(
                {
                    "paper_id": paper.get("id", ""),
                    "title": paper.get("title", ""),
                    "claims": claims,
                }
            )

        # Step 2: Generate novel hypotheses
        context = json.dumps(paper_claims, indent=2)[:8000]
        prompt = f"""You are a scientific hypothesis generation AI. Based on the following corpus of {len(papers)} papers in the domain of {domain}, generate {num_hypotheses} novel, testable hypotheses.

For each hypothesis provide:
- hypothesis_statement: A clear, declarative statement
- supporting_evidence: Which papers support this (list paper indices)
- contradictory_evidence: Which papers contradict or challenge this
- novelty_score: 0-1 how novel this is
- feasibility_score: 0-1 how feasible to test
- testability_score: 0-1 how testable
- suggested_experiment: How one could test this hypothesis

Corpus:
{context}

Return a JSON array of hypotheses."""
        result = LLMGateway.complete(
            model_id,
            [{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=4000,
        )

        try:
            hypotheses_data = json.loads(result.get("content", "[]"))
        except json.JSONDecodeError:
            content = result.get("content", "")
            try:
                start = content.index("[")
                end = content.rindex("]") + 1
                hypotheses_data = json.loads(content[start:end])
            except (ValueError, json.JSONDecodeError):
                hypotheses_data = []

        # Save to database
        saved_hypotheses = []
        for h in hypotheses_data:
            hypothesis = GeneratedHypothesis.objects.create(
                user_id=user_id,
                title=h.get("hypothesis_statement", "")[:200],
                domain=domain,
                hypothesis_statement=h.get("hypothesis_statement", ""),
                supporting_evidence=h.get("supporting_evidence", []),
                contradictory_evidence=h.get("contradictory_evidence", []),
                novelty_score=h.get("novelty_score", 0.5),
                feasibility_score=h.get("feasibility_score", 0.5),
                testability_score=h.get("testability_score", 0.5),
                suggested_experiment=h.get("suggested_experiment", ""),
                papers_analyzed_count=len(papers),
                generated_by_model=model_id,
                status="draft",
            )

            # Create source links
            supporting_indices = h.get("supporting_evidence", [])
            for idx in supporting_indices:
                if isinstance(idx, int) and idx < len(papers):
                    HypothesisSource.objects.create(
                        hypothesis=hypothesis,
                        paper_id=str(papers[idx].get("id", "")),
                        paper_title=papers[idx].get("title", ""),
                        contribution_type="direct_support",
                        relevance_score=0.8,
                    )

            saved_hypotheses.append(hypothesis)

        return saved_hypotheses

    @staticmethod
    def _extract_claims(paper: dict, model_id: str) -> list:
        from llm_hub.gateway import LLMGateway

        prompt = f"""Extract 3-5 key scientific claims from this paper abstract. Return a JSON list of strings.

Paper: {paper.get('title', '')}
Abstract: {paper.get('abstract', '')[:2000]}"""
        result = LLMGateway.complete(
            model_id, [{"role": "user", "content": prompt}], temperature=0.2
        )
        try:
            return json.loads(result.get("content", "[]"))
        except:
            return []
