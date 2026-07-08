"""AI Literature Synthesis & Structured Output Generator."""

import json
import logging

from django.utils import timezone

from .models import LiteratureReview

logger = logging.getLogger(__name__)


class LiteratureSynthesizer:
    """Generates structured literature reviews, slide decks, and infographics."""

    @staticmethod
    def synthesize(
        query: str, papers: list, output_type: str = "report", model_id: str = "gpt-4o"
    ) -> LiteratureReview:
        """Main synthesis pipeline."""

        review = LiteratureReview.objects.create(
            user_id=1,  # passed from view
            title=f"Review: {query[:100]}",
            query=query,
            papers_analyzed=len(papers),
            output_type=output_type,
            status="processing",
        )

        try:
            # Step 1: Extract structured data from each paper
            paper_summaries = []
            for paper in papers:
                summary = LiteratureSynthesizer._summarize_paper(paper, model_id)
                paper_summaries.append(summary)

            # Step 2: Generate the structured review
            if output_type == "report":
                content = LiteratureSynthesizer._generate_report(
                    query, paper_summaries, model_id
                )
                review.content_markdown = content
            elif output_type == "slides":
                slide_data = LiteratureSynthesizer._generate_slides(
                    query, paper_summaries, model_id
                )
                review.slide_deck_json = slide_data
                review.content_markdown = slide_data.get("summary", "")
            elif output_type == "infographic":
                infographic = LiteratureSynthesizer._generate_infographic(
                    query, paper_summaries, model_id
                )
                review.infographic_url = infographic.get("url", "")

            review.status = "completed"
            review.completed_at = timezone.now()
            review.source_papers = paper_summaries
            review.save()

        except Exception:
            logger.exception("Literature synthesis failed")
            review.status = "failed"
            review.save()

        return review

    @staticmethod
    def _summarize_paper(paper: dict, model_id: str) -> dict:
        from llm_hub.gateway import LLMGateway

        prompt = f"""Extract the following from this paper in JSON:
- key_findings (list of 3-5 bullet points)
- methodology (1-2 sentences)
- limitations (1-2 sentences)
- contribution_score (0-10)

Paper: {paper.get('title', '')}
Abstract: {paper.get('abstract', '')[:2000]}"""
        result = LLMGateway.complete(
            model_id, [{"role": "user", "content": prompt}], temperature=0.2
        )
        try:
            return json.loads(result.get("content", "{}"))
        except:
            return {
                "key_findings": [],
                "methodology": "",
                "limitations": "",
                "contribution_score": 5,
            }

    @staticmethod
    def _generate_report(query: str, paper_summaries: list, model_id: str) -> str:
        from llm_hub.gateway import LLMGateway

        context = "\n".join(
            f"Paper {i+1}: {s.get('key_findings', [])} | Method: {s.get('methodology', '')} | Score: {s.get('contribution_score', 5)}"
            for i, s in enumerate(paper_summaries)
        )
        prompt = f"""Write a comprehensive literature review on: {query}

Based on these paper summaries:
{context}

Structure the review with:
1. Executive Summary
2. Introduction
3. Key Themes & Findings
4. Methodological Approaches
5. Research Gaps
6. Future Directions
7. References

Write in academic style with proper citations."""
        result = LLMGateway.complete(
            model_id,
            [{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4000,
        )
        return result.get("content", "")

    @staticmethod
    def _generate_slides(query: str, paper_summaries: list, model_id: str) -> dict:
        from llm_hub.gateway import LLMGateway

        prompt = f"""Create a 10-slide presentation on: {query}
Based on {len(paper_summaries)} papers.
Return a JSON with keys: title, slides (array of {{title, bullets}}), summary."""
        result = LLMGateway.complete(
            model_id, [{"role": "user", "content": prompt}], temperature=0.3
        )
        try:
            return json.loads(result.get("content", "{}"))
        except:
            return {"title": query, "slides": [], "summary": ""}

    @staticmethod
    def _generate_infographic(query: str, paper_summaries: list, model_id: str) -> dict:
        return {"url": "", "status": "not_yet_implemented"}
