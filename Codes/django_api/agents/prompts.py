# agents/tests/test_prompts.py
"""
Prompt template test suite for the ML Research Assistant.
Run with: python manage.py test agents.tests.test_prompts
"""
from django.test import TestCase
from agents.tools import test_prompt_template

class PromptTemplateTests(TestCase):
    """Test that the agent's prompt templates produce expected outputs."""

    def test_experiment_config_template(self):
        """Test the experiment configuration prompt."""
        template = "Configure a {{model_name}} on {{task_name}} with {{method}}."
        test_cases = [
            {"model_name": "bert-base-uncased", "task_name": "mrpc", "method": "lora", "expected": "lora"},
            {"model_name": "distilbert", "task_name": "sst2", "method": "baseline", "expected": "baseline"},
        ]
        result = test_prompt_template({"template": template, "test_cases": test_cases})
        self.assertEqual(len(result["results"]), 2)
        self.assertIn("lora", result["results"][0]["substituted_prompt"])
        self.assertIn("baseline", result["results"][1]["substituted_prompt"])

    def test_latex_edit_template(self):
        """Test the LaTeX editing prompt."""
        template = "Edit file {{filename}} in project {{project_id}}."
        test_cases = [
            {"filename": "main.tex", "project_id": "abc-123", "expected": "main.tex"},
        ]
        result = test_prompt_template({"template": template, "test_cases": test_cases})
        self.assertIn("abc-123", result["results"][0]["substituted_prompt"])

    def test_analysis_template(self):
        """Test training analysis prompt."""
        template = "Analyze run {{run_id}} for overfitting and plateau."
        test_cases = [
            {"run_id": "a1b2c3d4", "expected": "a1b2c3d4"},
        ]
        result = test_prompt_template({"template": template, "test_cases": test_cases})
        self.assertIn("a1b2c3d4", result["results"][0]["substituted_prompt"])