"""
Impact scoring engine.

Currently uses a weighted formula as a placeholder.
Replace the `predict()` function with a real ML model later.
"""

import numpy as np


def extract_features(project):
    """
    Extract the feature vector from a ResearchProject instance.
    Returns a dict of normalized scores for each factor.
    """
    features = {}

    # 1. Institution quality (0–100)
    inst_score = 50.0  # baseline
    if project.institution_age_years:
        inst_score += min(project.institution_age_years, 50) * 0.5
    if project.institution_location:
        # Simplified: assign region weights (can be replaced with a lookup)
        location = project.institution_location.lower()
        if any(
            x in location
            for x in ["stanford", "mit", "cambridge", "oxford", "berkeley"]
        ):
            inst_score += 25
        elif any(x in location for x in ["usa", "uk", "germany", "canada"]):
            inst_score += 10
    features["institution_score"] = min(inst_score, 100)

    # 2. Investigator quality (0–100)
    inv_score = 30.0
    if project.investigator_h_index:
        inv_score += min(project.investigator_h_index, 100) * 0.4
    if project.investigator_publications:
        inv_score += min(project.investigator_publications, 200) * 0.1
    if project.investigator_previous_grants:
        inv_score += min(project.investigator_previous_grants, 10) * 2.0
    features["investigator_score"] = min(inv_score, 100)

    # 3. Digital traction (0–100)
    digi_score = 10.0
    if project.social_followers:
        digi_score += min(project.social_followers, 10000) * 0.005
    if project.preprint_views:
        digi_score += min(project.preprint_views, 5000) * 0.01
    if project.online_mentions:
        digi_score += min(project.online_mentions, 100) * 0.3
    features["digital_traction"] = min(digi_score, 100)

    # 4. Funding history (0–100)
    fund_score = 20.0
    if project.previous_funding_total:
        fund_score += min(float(project.previous_funding_total), 1000000) * 0.00005
    if project.previous_grants_count:
        fund_score += min(project.previous_grants_count, 20) * 2.0
    features["funding_history"] = min(fund_score, 100)

    return features


def predict(features):
    """
    Predict impact score (0–100) from extracted features.

    Weights based on the initial requirements:
    - Institution quality: 20%
    - Investigator quality: 35%
    - Digital traction: 15%
    - Funding history: 30%
    """
    weights = {
        "institution_score": 0.20,
        "investigator_score": 0.35,
        "digital_traction": 0.15,
        "funding_history": 0.30,
    }

    score = sum(features[k] * weights[k] for k in weights)
    score = min(max(score, 0), 100)

    # Confidence interval: ±8% of score, clamped to 0–100
    margin = 8.0
    lower = max(score - margin, 0)
    upper = min(score + margin, 100)

    # Percentile: map score to a rough percentile (0–100 → 0–100)
    percentile = max(1, min(99, score * 0.85 + 10))

    # Funding probability (simple sigmoid-like mapping)
    funding_prob = round(1.0 / (1.0 + np.exp(-(score - 50) / 15)), 4)

    # Publication tier
    if score >= 85:
        tier = "Top 5%"
    elif score >= 65:
        tier = "Top 15%"
    elif score >= 40:
        tier = "Top 35%"
    else:
        tier = "Average"

    return {
        "impact_score": round(score, 2),
        "confidence_lower": round(lower, 2),
        "confidence_upper": round(upper, 2),
        "percentile": round(percentile, 2),
        "funding_probability": funding_prob,
        "publication_tier": tier,
        "comparison_text": f"Top {int(100 - percentile)}% of similar projects",
        "factor_scores": features,
    }
