ROUTER_KEYWORDS = {
    'comparative': ['better', 'worse', 'than', 'more', 'less', 'higher', 'lower', 'superior', 'inferior', 'best', 'worst', 'prefer'],
    'negation': ['not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nor', 'without', "n't", 'dont', "don't", 'wont', "won't", 'cant', "can't"],
    'irony': []
}

def detect_task_type(text: str) -> str:
    text_lower = text.lower()
    comp_count = sum(1 for kw in ROUTER_KEYWORDS['comparative'] if kw in text_lower)
    neg_count = sum(1 for kw in ROUTER_KEYWORDS['negation'] if kw in text_lower)
    if comp_count > neg_count and comp_count > 0:
        return 'comparative'
    elif neg_count > 0:
        return 'negation'
    return 'irony'