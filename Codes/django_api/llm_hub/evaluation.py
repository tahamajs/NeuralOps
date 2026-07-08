import json
import logging

from django.utils import timezone

from .gateway import LLMGateway
from .models import EvalResult, EvalRun

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, run: EvalRun):
        self.run = run
        self.model_id = run.model.model_id

    def execute(self):
        self.run.status = "running"
        self.run.started_at = timezone.now()
        self.run.save()
        try:
            results = []
            for test_case in self.run.suite.test_cases.all():
                output = LLMGateway.complete(
                    self.model_id,
                    messages=[{"role": "user", "content": test_case.input_text}],
                    temperature=0.0,
                )
                result = EvalResult.objects.create(
                    run=self.run,
                    test_case=test_case,
                    model_output=output.get("content", ""),
                )
                score, passed = self._evaluate(test_case, output.get("content", ""))
                result.score = score
                result.passed = passed
                result.save()
                results.append(result)

            scores = [r.score for r in results if r.score is not None]
            self.run.overall_score = sum(scores) / len(scores) if scores else 0
            self.run.status = "completed"
        except Exception:
            self.run.status = "failed"
            logger.exception("Evaluation failed")
        finally:
            self.run.completed_at = timezone.now()
            self.run.save()

    def _evaluate(self, test_case, model_output):
        eval_type = test_case.evaluation_type
        expected = test_case.expected_output

        if eval_type == "exact_match":
            passed = model_output.strip() == expected.strip()
            return (1.0 if passed else 0.0), passed
        elif eval_type == "contains":
            passed = expected.lower() in model_output.lower()
            return (1.0 if passed else 0.0), passed
        elif eval_type == "semantic_similarity":
            try:
                from sentence_transformers import SentenceTransformer
                from sklearn.metrics.pairwise import cosine_similarity

                model = SentenceTransformer("all-MiniLM-L6-v2")
                emb1 = model.encode([expected])[0]
                emb2 = model.encode([model_output])[0]
                sim = cosine_similarity([emb1], [emb2])[0][0]
                passed = sim > 0.8
                return float(sim), passed
            except ImportError:
                return 0.5, False
        elif eval_type == "llm_judge":
            judge_prompt = f"""Score the following model response on a scale of 0-10 based on how well it matches the expected output.
Expected: {expected}
Model Response: {model_output}
Return only a JSON: {{"score": float, "reason": "..."}}"""
            judge_result = LLMGateway.complete(
                "gpt-4o",
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0,
            )
            try:
                judge_json = json.loads(judge_result.get("content", "{}"))
                score = float(judge_json.get("score", 0)) / 10.0
                passed = score >= 0.7
                return score, passed
            except:
                return 0.5, False
        elif eval_type == "tool_call_correct":
            try:
                expected_tool = json.loads(expected)
                model_tool = json.loads(model_output)
                passed = expected_tool == model_tool
                return (1.0 if passed else 0.0), passed
            except:
                return 0.0, False
        return 0.0, False
