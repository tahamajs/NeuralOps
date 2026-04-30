# framework/routing/keyword_router.py
"""
Keyword-based task detection router.

Routes inputs based on keyword presence in text.
"""

from typing import Dict, List, Tuple, Optional
import re


# Comprehensive keyword database
ROUTER_KEYWORDS: Dict[str, List[str]] = {
    'comparative': [
        'better', 'worse', 'than', 'more', 'less', 'higher', 'lower', 
        'superior', 'inferior', 'best', 'worst', 'prefer', 'preferable',
        'rather', 'instead', 'compared', 'comparison', 'versus', 'vs',
        'outperform', 'exceed', 'surpass', 'behind', 'ahead', 'above',
        'below', 'least', 'most', 'greater', 'smaller', 'faster', 'slower',
        'cheaper', 'expensive', 'stronger', 'weaker', 'bigger', 'smaller'
    ],
    'negation': [
        'not', 'no', 'never', 'nobody', 'nothing', 'neither', 'nor', 
        'without', "n't", 'dont', "don't", 'wont', "won't", 'cant', "can't",
        'shouldnt', "shouldn't", 'wouldnt', "wouldn't", 'couldnt', "couldn't",
        'didnt', "didn't", 'hasnt', "hasn't", 'haven\'t', 'isnt', "isn't",
        'arent', "aren't", 'wasnt', "wasn't", 'werent', "weren't", 'no one',
        'none', 'nowhere', 'hardly', 'barely', 'scarcely', 'rarely'
    ],
    'sentiment': [
        'good', 'great', 'awesome', 'amazing', 'wonderful', 'excellent',
        'fantastic', 'terrible', 'awful', 'horrible', 'bad', 'poor',
        'love', 'like', 'hate', 'dislike', 'enjoy', 'happy', 'sad',
        'angry', 'frustrated', 'disappointed', 'satisfied', 'pleased'
    ],
    'question': [
        'what', 'why', 'how', 'when', 'where', 'who', 'which', 'whom',
        'does', 'do', 'is', 'are', 'was', 'were', 'have', 'has', 'can',
        'could', 'would', 'should', 'will', 'may', 'might', '?'
    ],
    'irony': [
        'ironically', 'sarcastically', 'sarcasm', 'irony', 'unfortunately',
        'ironic', 'of course', 'as if', 'yeah right', 'sure', 'whatever'
    ],
    'factual': [
        'according', 'research', 'study', 'shows', 'demonstrates',
        'evidence', 'data', 'statistics', 'results', 'findings',
        'published', 'report', 'analysis', 'survey'
    ],
    'opinion': [
        'think', 'believe', 'feel', 'opinion', 'view', 'perspective',
        'personally', 'in my opinion', 'i think', 'i believe', 'i feel'
    ]
}


class KeywordRouter:
    """
    Keyword-based router for detecting task types.
    
    Uses weighted keyword matching to determine the most appropriate
    task category for a given input text.
    
    Example:
        >>> router = KeywordRouter()
        >>> task = router.route("This product is better than the other one")
        >>> print(task)  # 'comparative'
    """
    
    def __init__(
        self, 
        keywords: Optional[Dict[str, List[str]]] = None,
        case_sensitive: bool = False,
        use_weights: bool = True
    ):
        """
        Args:
            keywords: Custom keyword dictionary (uses default if None)
            case_sensitive: Whether to match case sensitively
            use_weights: Whether to use weighted scoring
        """
        self.keywords = keywords or ROUTER_KEYWORDS
        self.case_sensitive = case_sensitive
        self.use_weights = use_weights
        
        # Precompile regex patterns for speed
        self._patterns = {}
        for task_type, words in self.keywords.items():
            if words:
                pattern = r'\b(' + '|'.join(re.escape(w) for w in words) + r')\b'
                flags = 0 if case_sensitive else re.IGNORECASE
                self._patterns[task_type] = re.compile(pattern, flags)
    
    def route(self, text: str) -> Tuple[str, Dict[str, int]]:
        """
        Route text to a task type.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (detected_task_type, confidence_scores)
            
        Example:
            >>> task_type, scores = router.route("This is better than that")
            >>> print(task_type)  # 'comparative'
        """
        text_lower = text if self.case_sensitive else text.lower()
        scores = {}
        
        for task_type, words in self.keywords.items():
            if self.use_weights:
                # Weighted scoring: longer matches count more
                score = 0
                for word in words:
                    count = text_lower.count(word.lower() if not self.case_sensitive else word)
                    # Longer words get higher weight
                    weight = len(word) / 10
                    score += count * (weight if weight > 0.5 else 0.5)
                scores[task_type] = score
            else:
                # Simple counting
                count = sum(1 for word in words if word in text_lower)
                scores[task_type] = count
        
        # Find best match
        best_task = max(scores, key=scores.get) if any(scores.values()) else 'factual'
        return best_task, scores
    
    def batch_route(self, texts: List[str]) -> List[str]:
        """
        Route multiple texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            List of detected task types
        """
        return [self.route(text)[0] for text in texts]
    
    def get_confidence(self, text: str) -> float:
        """
        Get confidence score for the best match.
        
        Args:
            text: Input text
            
        Returns:
            Confidence score between 0 and 1
        """
        task, scores = self.route(text)
        total = sum(scores.values())
        if total == 0:
            return 0.0
        return scores[task] / total
    
    def add_keywords(self, task_type: str, keywords: List[str]):
        """Add new keywords for a task type."""
        if task_type not in self.keywords:
            self.keywords[task_type] = []
        self.keywords[task_type].extend(keywords)
        
        # Recompile pattern
        pattern = r'\b(' + '|'.join(re.escape(w) for w in self.keywords[task_type]) + r')\b'
        flags = 0 if self.case_sensitive else re.IGNORECASE
        self._patterns[task_type] = re.compile(pattern, flags)
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics about keyword coverage."""
        return {task: len(words) for task, words in self.keywords.items()}


# Convenience function
def detect_task_type(text: str) -> str:
    """
    Simple convenience function for keyword-based task detection.
    
    Args:
        text: Input text
        
    Returns:
        Detected task type ('comparative', 'negation', 'sentiment', etc.)
    """
    router = KeywordRouter()
    return router.route(text)[0]