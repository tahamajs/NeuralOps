"""
AI Agent for natural language query parsing and filtering.
Extracts structured filters from free‑text queries.
"""
import re
import json
import logging

logger = logging.getLogger(__name__)

# Simple rule‑based parser (can be replaced with an LLM call)
GPU_MODEL_PATTERNS = [
    r'\b(A100|H100|H200|B200|RTX\s?\d{4}|V100|T4|A6000|A5000|A4000|P100|K80)\b',
]
PRICE_PATTERNS = [
    r'(?:under|less than|below|max|maximum|≤|<=)\s*\$?(\d+\.?\d*)\s*(?:\/hr|per hour)?',
    r'\$(\d+\.?\d*)\s*(?:\/hr|per hour)',
]
VRAM_PATTERNS = [
    r'(\d+)\s*(?:GB|gb)\s*(?:VRAM|memory|vram)',
    r'(?:at least|minimum|min|≥|>=)\s*(\d+)\s*(?:GB|gb)',
]


def parse_query(query_text: str) -> dict:
    """
    Parse a natural language query into structured filters.
    Returns a dict of filters that can be passed to the listing search API.
    """
    filters = {}

    # Extract GPU model
    for pattern in GPU_MODEL_PATTERNS:
        match = re.search(pattern, query_text, re.IGNORECASE)
        if match:
            filters['gpu_model'] = match.group(1).replace(' ', '')
            break

    # Extract price
    for pattern in PRICE_PATTERNS:
        match = re.search(pattern, query_text, re.IGNORECASE)
        if match:
            filters['max_price'] = float(match.group(1))
            break

    # Extract VRAM
    for pattern in VRAM_PATTERNS:
        match = re.search(pattern, query_text, re.IGNORECASE)
        if match:
            filters['min_vram'] = int(match.group(1))
            break

    # Extract listing type
    if re.search(r'\b(spot|interruptible)\b', query_text, re.IGNORECASE):
        filters['listing_type'] = 'spot'
    elif re.search(r'\b(reserved|long.term)\b', query_text, re.IGNORECASE):
        filters['listing_type'] = 'reserved'
    elif re.search(r'\b(on.demand|guaranteed)\b', query_text, re.IGNORECASE):
        filters['listing_type'] = 'on_demand'

    # Extract sort preference
    if re.search(r'\b(cheapest|lowest price|affordable)\b', query_text, re.IGNORECASE):
        filters['sort'] = 'price_asc'
    elif re.search(r'\b(best|highest rated|most reliable)\b', query_text, re.IGNORECASE):
        filters['sort'] = 'reliability'
    elif re.search(r'\b(fastest|best performance|dlperf)\b', query_text, re.IGNORECASE):
        filters['sort'] = 'dlperf'

    return filters


def generate_explanation(filters: dict, listing) -> str:
    """
    Generate a human‑readable explanation of why a listing matches a query.
    """
    reasons = []
    if 'gpu_model' in filters:
        if filters['gpu_model'].lower() in listing.gpu_model.lower():
            reasons.append(f"✅ Matches requested model: {filters['gpu_model']}")
        else:
            reasons.append(f"⚠️  Different model than requested ({listing.gpu_model})")
    if 'max_price' in filters:
        if float(listing.current_price_per_hour) <= filters['max_price']:
            reasons.append(f"✅ Within budget: ${listing.current_price_per_hour}/hr")
        else:
            reasons.append(f"❌ Over budget: ${listing.current_price_per_hour}/hr")
    if 'min_vram' in filters:
        if listing.vram_gb >= filters['min_vram']:
            reasons.append(f"✅ Sufficient VRAM: {listing.vram_gb} GB")
        else:
            reasons.append(f"❌ Insufficient VRAM: {listing.vram_gb} GB")

    return '\n'.join(reasons) if reasons else "Matches your search criteria"


def check_compliance(listing, requirements: dict) -> dict:
    """
    Check if a GPU listing complies with the user's requirements.
    Returns {passed: bool, failures: list, warnings: list}.
    """
    failures = []
    warnings = []

    if 'min_vram' in requirements:
        if listing.vram_gb < requirements['min_vram']:
            failures.append(f"VRAM: {listing.vram_gb} GB < required {requirements['min_vram']} GB")
        elif listing.vram_gb < requirements['min_vram'] * 1.2:
            warnings.append(f"VRAM: {listing.vram_gb} GB is close to minimum {requirements['min_vram']} GB")

    if 'max_price' in requirements:
        if float(listing.current_price_per_hour) > requirements['max_price']:
            failures.append(f"Price: ${listing.current_price_per_hour}/hr > max ${requirements['max_price']}/hr")
        elif float(listing.current_price_per_hour) > requirements['max_price'] * 0.8:
            warnings.append(f"Price: ${listing.current_price_per_hour}/hr is close to max ${requirements['max_price']}/hr")

    if 'min_reliability' in requirements:
        if listing.reliability_score < requirements['min_reliability']:
            failures.append(f"Reliability: {listing.reliability_score} < required {requirements['min_reliability']}")

    if 'min_dlperf' in requirements:
        if (listing.dlperf_score or 0) < requirements['min_dlperf']:
            failures.append(f"DLPerf: {listing.dlperf_score} < required {requirements['min_dlperf']}")

    if 'location' in requirements:
        if requirements['location'].lower() not in (listing.location or '').lower():
            failures.append(f"Location: {listing.location} does not match {requirements['location']}")

    return {
        'passed': len(failures) == 0,
        'failures': failures,
        'warnings': warnings,
    }