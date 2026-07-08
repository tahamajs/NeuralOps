"""
Core forensic engine – analyzes logs, classifies failures, and suggests fixes.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

# Pre‑defined error patterns with regex and suggestions
BUILTIN_PATTERNS = [
    {
        "pattern_name": "CUDA Out of Memory",
        "regex": r"CUDA out of memory|RuntimeError.*out of memory|torch.cuda.OutOfMemoryError",
        "root_cause_category": "out_of_memory",
        "description": "GPU ran out of memory during training.",
        "suggestion": "Reduce batch size, enable gradient accumulation, or use mixed precision training.",
    },
    {
        "pattern_name": "NaN Loss",
        "regex": r"NaN.*loss|Loss is NaN|loss.*=.*nan",
        "root_cause_category": "numerical_error",
        "description": "Loss function returned NaN, likely due to exploding gradients or bad data.",
        "suggestion": "Add gradient clipping, reduce learning rate, or check for corrupt data.",
    },
    {
        "pattern_name": "CUDA Device Not Found",
        "regex": r"CUDA.*not available|No CUDA|device.*not found|cuda.*not detected",
        "root_cause_category": "infrastructure_issue",
        "description": "CUDA device is not accessible or not detected.",
        "suggestion": "Verify GPU drivers are installed and CUDA toolkit is properly configured.",
    },
    {
        "pattern_name": "ModuleNotFoundError",
        "regex": r"ModuleNotFoundError|ImportError|No module named",
        "root_cause_category": "dependency_conflict",
        "description": "A required Python package is missing.",
        "suggestion": "Run: pip install <missing-package>. Check requirements.txt.",
    },
    {
        "pattern_name": "Data Shape Mismatch",
        "regex": r"shape.*mismatch|size.*mismatch|expected.*size|mat1 and mat2 shapes",
        "root_cause_category": "data_mismatch",
        "description": "Tensor shape mismatch, likely due to incorrect data preprocessing.",
        "suggestion": "Verify input data shapes and model architecture dimensions.",
    },
    {
        "pattern_name": "Network Timeout",
        "regex": r"ConnectionError|Timeout|HTTP.*error|Connection refused",
        "root_cause_category": "network_issue",
        "description": "Network connection failed, possibly when downloading data or models.",
        "suggestion": "Check network connectivity and retry. Consider caching downloads locally.",
    },
    {
        "pattern_name": "Disk Full",
        "regex": r"No space left on device|Disk quota exceeded",
        "root_cause_category": "infrastructure_issue",
        "description": "Disk is full – cannot save checkpoints or logs.",
        "suggestion": "Free up disk space or increase storage allocation.",
    },
]


class FailureForensicEngine:
    """Analyzes training run failures and generates forensic reports."""

    def __init__(self, run):
        self.run = run
        self.logs = run.logs or ""
        self.config = run.config or {}
        self.metrics = run.final_metrics or {}

    def analyze(self) -> Dict:
        """Run the full forensic analysis pipeline."""
        # 1. Extract error lines from logs
        error_lines = self._extract_error_lines()

        # 2. Match against known patterns
        matched_patterns = self._match_patterns(error_lines)

        # 3. Classify root cause
        root_cause = self._classify_root_cause(matched_patterns, error_lines)

        # 4. Generate suggestions
        suggestions = self._generate_suggestions(matched_patterns, root_cause)

        # 5. Create fix actions
        fix_actions = self._create_fix_actions(root_cause, suggestions)

        return {
            "root_cause_category": root_cause["category"],
            "root_cause_detail": root_cause["detail"],
            "confidence": root_cause["confidence"],
            "error_logs": error_lines[:20],  # keep the most relevant
            "anomaly_patterns": matched_patterns,
            "suggestions": suggestions,
            "fix_actions": fix_actions,
        }

    def _extract_error_lines(self) -> List[str]:
        """Extract lines that contain error keywords."""
        keywords = [
            "error",
            "Error",
            "ERROR",
            "fail",
            "Fail",
            "FAIL",
            "traceback",
            "Traceback",
            "exception",
            "Exception",
            "nan",
            "NaN",
            "NAN",
            "inf",
            "Inf",
            "killed",
        ]
        lines = []
        for line in self.logs.split("\n"):
            if any(kw in line for kw in keywords):
                lines.append(line.strip()[:500])
        return lines[-50:]  # last 50 error lines are most relevant

    def _match_patterns(self, error_lines: List[str]) -> List[Dict]:
        """Match error lines against built‑in and saved patterns."""
        matched = []
        for pattern in BUILTIN_PATTERNS:
            regex = re.compile(pattern["regex"], re.IGNORECASE)
            matching_lines = [line for line in error_lines if regex.search(line)]
            if matching_lines:
                matched.append(
                    {
                        "pattern_name": pattern["pattern_name"],
                        "root_cause_category": pattern["root_cause_category"],
                        "description": pattern["description"],
                        "suggestion": pattern["suggestion"],
                        "matching_lines": matching_lines[:5],
                    }
                )

        # Also check custom saved patterns
        from .models import ErrorPattern

        custom_patterns = ErrorPattern.objects.filter(is_active=True)
        for cp in custom_patterns:
            regex = re.compile(cp.regex_pattern, re.IGNORECASE)
            matching_lines = [line for line in error_lines if regex.search(line)]
            if matching_lines:
                matched.append(
                    {
                        "pattern_name": cp.pattern_name,
                        "root_cause_category": cp.root_cause_category,
                        "description": cp.description,
                        "suggestion": cp.suggestion,
                        "matching_lines": matching_lines[:5],
                    }
                )
                cp.occurrences += 1
                cp.save(update_fields=["occurrences"])

        return matched

    def _classify_root_cause(
        self, matched_patterns: List[Dict], error_lines: List[str]
    ) -> Dict:
        """Classify the most likely root cause."""
        if not matched_patterns and not error_lines:
            return {
                "category": "unknown",
                "detail": "No errors found in logs.",
                "confidence": 0.0,
            }

        if not matched_patterns:
            # No known patterns matched – do basic heuristics
            return self._heuristic_classify(error_lines)

        # Use the pattern with the most matching lines as primary
        best = max(matched_patterns, key=lambda p: len(p["matching_lines"]))
        confidence = min(0.95, 0.5 + len(best["matching_lines"]) * 0.1)
        return {
            "category": best["root_cause_category"],
            "detail": best["description"],
            "confidence": round(confidence, 2),
        }

    def _heuristic_classify(self, error_lines: List[str]) -> Dict:
        """Basic heuristics when no pattern matches."""
        text = " ".join(error_lines).lower()
        if "cuda" in text or "gpu" in text:
            return {
                "category": "infrastructure_issue",
                "detail": "GPU or CUDA‑related error detected.",
                "confidence": 0.4,
            }
        if "memory" in text or "ram" in text:
            return {
                "category": "out_of_memory",
                "detail": "Memory‑related error detected.",
                "confidence": 0.4,
            }
        if "shape" in text or "dimension" in text:
            return {
                "category": "data_mismatch",
                "detail": "Shape or dimension mismatch detected.",
                "confidence": 0.4,
            }
        return {
            "category": "unknown",
            "detail": "Could not classify the failure automatically.",
            "confidence": 0.1,
        }

    def _generate_suggestions(
        self, matched_patterns: List[Dict], root_cause: Dict
    ) -> List[str]:
        """Generate human‑readable suggestions."""
        suggestions = []
        for pattern in matched_patterns:
            if pattern["suggestion"] not in suggestions:
                suggestions.append(pattern["suggestion"])
        if not suggestions:
            suggestions.append(
                "Review the full logs manually. Check common issues: "
                "batch size, learning rate, data shapes, and GPU memory."
            )
        return suggestions[:5]

    def _create_fix_actions(
        self, root_cause: Dict, suggestions: List[str]
    ) -> List[Dict]:
        """Map root cause to concrete one‑click fix actions."""
        action_map = {
            "out_of_memory": [
                {
                    "action_type": "adjust_config",
                    "description": "Reduce batch size by 50%.",
                    "config_changes": {"batch_size": "auto_reduce"},
                },
                {
                    "action_type": "increase_resources",
                    "description": "Request a GPU with more VRAM.",
                    "config_changes": {},
                },
            ],
            "numerical_error": [
                {
                    "action_type": "adjust_config",
                    "description": "Add gradient clipping (max_norm=1.0) and reduce learning rate by 10x.",
                    "config_changes": {
                        "learning_rate": "auto_reduce",
                        "gradient_clipping": 1.0,
                    },
                },
            ],
            "configuration_error": [
                {
                    "action_type": "adjust_config",
                    "description": "Restore default configuration and retry.",
                    "config_changes": {"restore_defaults": True},
                },
            ],
            "dependency_conflict": [
                {
                    "action_type": "reinstall_deps",
                    "description": "Reinstall dependencies from requirements.txt.",
                    "config_changes": {},
                },
            ],
            "infrastructure_issue": [
                {
                    "action_type": "restart",
                    "description": "Restart the training job after verifying hardware.",
                    "config_changes": {},
                },
            ],
            "data_mismatch": [
                {
                    "action_type": "adjust_config",
                    "description": "Verify and correct input dimensions.",
                    "config_changes": {"validate_data": True},
                },
            ],
            "network_issue": [
                {
                    "action_type": "restart",
                    "description": "Retry the job; network issues are often transient.",
                    "config_changes": {},
                },
            ],
        }
        actions = action_map.get(
            root_cause["category"],
            [
                {
                    "action_type": "restart",
                    "description": "Restart the training job.",
                    "config_changes": {},
                },
            ],
        )

        for suggestion in suggestions:
            if suggestion not in [a["description"] for a in actions]:
                actions.append(
                    {
                        "action_type": "adjust_config",
                        "description": suggestion,
                        "config_changes": {},
                    }
                )

        return actions
