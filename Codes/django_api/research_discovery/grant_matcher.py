"""AI Grant Discovery & Proposal Matching Engine."""

import json
import logging

from .models import GrantMatch, GrantOpportunity, ResearcherGrantProfile

logger = logging.getLogger(__name__)


class GrantDiscoveryEngine:
    """Searches global grant databases and matches researchers to opportunities."""

    GRANT_SOURCES = {
        "nih": {
            "name": "NIH RePORTER",
            "api": "https://api.reporter.nih.gov/v2/projects/search",
            "method": "POST",
        },
        "nsf": {
            "name": "NSF Awards",
            "api": "https://api.nsf.gov/services/v1/awards.json",
            "method": "GET",
        },
        "europe": {
            "name": "EU Funding & Tenders",
            "api": "https://ec.europa.eu/info/funding-tenders/opportunities/api",
            "method": "GET",
        },
        "wellcome": {
            "name": "Wellcome Trust",
            "api": "https://wellcome.org/grant-funding/api",
            "method": "GET",
        },
    }

    @classmethod
    def search_grants(cls, keywords: list, limit: int = 20) -> list:
        """Search for grant opportunities from multiple sources."""
        results = []

        # NIH RePORTER
        try:
            nih_results = cls._search_nih(keywords, limit)
            results.extend(nih_results)
        except Exception as e:
            logger.warning("NIH search failed: %s", e)

        # NSF
        try:
            nsf_results = cls._search_nsf(keywords, limit)
            results.extend(nsf_results)
        except Exception as e:
            logger.warning("NSF search failed: %s", e)

        # Save new grants to database
        for grant_data in results:
            GrantOpportunity.objects.get_or_create(
                source=grant_data["source"],
                source_id=grant_data.get("source_id", ""),
                defaults={
                    "title": grant_data["title"],
                    "description": grant_data.get("description", ""),
                    "funding_amount": grant_data.get("amount", ""),
                    "deadline": grant_data.get("deadline"),
                    "url": grant_data.get("url", ""),
                    "keywords": keywords,
                },
            )

        return results

    @classmethod
    def match_researcher(cls, user_id: int, model_id: str = "gpt-4o") -> list:
        """Match a researcher profile against active grant opportunities."""
        from llm_hub.gateway import LLMGateway

        profile = ResearcherGrantProfile.objects.get_or_create(user_id=user_id)[0]
        grants = GrantOpportunity.objects.filter(is_active=True)

        if not grants.exists():
            return []

        # Build researcher context
        profile_text = f"""
Research keywords: {json.dumps(profile.research_keywords)}
Publications: {json.dumps(profile.publications[:5])}
Career stage: {profile.career_stage}
Institution: {profile.institution}"""

        matches = []
        for grant in grants[:30]:
            prompt = f"""Rate how well this grant matches the researcher on a scale of 0-1.
Researcher: {profile_text}
Grant: {grant.title}
Description: {grant.description or ''}
Return JSON: {{"score": float, "reasons": ["..."], "suggested_proposal_title": "..."}}"""
            result = LLMGateway.complete(
                model_id, [{"role": "user", "content": prompt}], temperature=0.1
            )
            try:
                match_data = json.loads(result.get("content", "{}"))
                score = float(match_data.get("score", 0))
                if score > 0.3:
                    grant_match = GrantMatch.objects.create(
                        user_id=user_id,
                        grant=grant,
                        match_score=score,
                        match_reasons=match_data.get("reasons", []),
                        suggested_proposal_title=match_data.get(
                            "suggested_proposal_title", ""
                        ),
                    )
                    matches.append(grant_match)
            except (json.JSONDecodeError, ValueError):
                continue

        return sorted(matches, key=lambda m: m.match_score, reverse=True)

    @classmethod
    def _search_nih(cls, keywords: list, limit: int) -> list:
        import requests

        resp = requests.post(
            cls.GRANT_SOURCES["nih"]["api"],
            json={
                "criteria": {
                    "advanced_text_search": {
                        "operator": "and",
                        "search_text": " OR ".join(keywords),
                    }
                },
                "offset": 0,
                "limit": limit,
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "source": "nih",
                "source_id": str(r.get("project_num", "")),
                "title": r.get("project_title", ""),
                "description": r.get("abstract_text", "")[:500],
                "amount": f"${r.get('award_amount', 0):,}",
                "url": f"https://reporter.nih.gov/project-details/{r.get('project_num', '')}",
            }
            for r in data.get("results", [])
        ]

    @classmethod
    def _search_nsf(cls, keywords: list, limit: int) -> list:
        import requests

        resp = requests.get(
            cls.GRANT_SOURCES["nsf"]["api"],
            params={
                "keyword": " OR ".join(keywords),
                "printFields": "title,abstractText,awardAmount",
                "status": "active",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            {
                "source": "nsf",
                "source_id": str(a.get("id", "")),
                "title": a.get("title", ""),
                "description": a.get("abstractText", "")[:500],
                "amount": f"${a.get('awardAmount', 0):,}",
                "url": f"https://www.nsf.gov/awardsearch/showAward?AWD_ID={a.get('id', '')}",
            }
            for a in data.get("response", {}).get("award", [])[:limit]
        ]
