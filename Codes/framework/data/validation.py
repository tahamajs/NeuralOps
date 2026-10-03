"""
Simple data validation – missing values, infinite, zero variance, schema checks.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class ValidationReport:
    checks_passed: int
    checks_failed: int
    issues: list[dict[str, Any]]
    is_valid: bool


class DataValidator:
    def __init__(self, schema: dict[str, Any] = None) -> None:
        self.schema = schema or {}

    def validate(self, dataset: dict[str, np.ndarray]) -> ValidationReport:
        """Validate."""
        issues = []
        for col, values in dataset.items():
            v = np.array(values)
            # Missing values
            missing = (
                np.sum(np.isnan(v))
                if np.issubdtype(v.dtype, np.floating)
                else np.sum(v is None)
            )
            if missing > 0:
                issues.append(
                    {"column": col, "issue": "missing_values", "count": int(missing)}
                )
            # Infinite values
            if v.dtype in [np.float32, np.float64]:
                inf_count = np.sum(np.isinf(v))
                if inf_count > 0:
                    issues.append(
                        {
                            "column": col,
                            "issue": "infinite_values",
                            "count": int(inf_count),
                        }
                    )
            # Low variance
            if len(v) > 1 and v.dtype in [np.float32, np.float64]:
                if np.std(v) < 1e-8:
                    issues.append({"column": col, "issue": "zero_variance"})
            # Schema checks
            if col in self.schema:
                expected = self.schema[col]
                if "min" in expected and np.min(v) < expected["min"]:
                    issues.append(
                        {
                            "column": col,
                            "issue": "below_min",
                            "min": expected["min"],
                            "actual": float(np.min(v)),
                        }
                    )
                if "max" in expected and np.max(v) > expected["max"]:
                    issues.append(
                        {
                            "column": col,
                            "issue": "above_max",
                            "max": expected["max"],
                            "actual": float(np.max(v)),
                        }
                    )

        failed_cols = set(i["column"] for i in issues)
        return ValidationReport(
            checks_passed=len(dataset) - len(failed_cols),
            checks_failed=len(failed_cols),
            issues=issues,
            is_valid=len(issues) == 0,
        )
