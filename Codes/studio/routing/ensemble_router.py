# studio/routing/ensemble_router.py
class EnsembleRouter:
    def __init__(self, routers, weights=None):
        self.routers = routers
        self.weights = weights or [1.0/len(routers)]*len(routers)
    def route(self, text):
        scores = {}
        for router, w in zip(self.routers, self.weights):
            pred, probs = router.route(text)
            for cls, p in probs.items():
                scores[cls] = scores.get(cls, 0) + p * w
        best = max(scores, key=scores.get)
        return best, scores