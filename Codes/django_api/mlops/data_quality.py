"""Automated data quality checks (pydeequ‑style)."""
import pandas as pd
from .models import DataQualityRule, DataQualityCheck

class DataQualityRunner:
    @staticmethod
    def run_check(rule: DataQualityRule, dataset_path: str) -> DataQualityCheck:
        df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.DataFrame()
        config = rule.config
        passed = True
        failures = 0
        details = {}

        if rule.rule_type == 'completeness':
            col = config['column']
            missing = df[col].isnull().sum()
            failures = missing
            passed = missing == 0
            details = {'missing_count': int(missing), 'total': len(df)}

        elif rule.rule_type == 'range':
            col, min_val, max_val = config['column'], config.get('min'), config.get('max')
            out_of_range = df[(df[col] < min_val) | (df[col] > max_val)].shape[0]
            failures = out_of_range
            passed = out_of_range == 0
            details = {'out_of_range': int(out_of_range)}

        elif rule.rule_type == 'schema':
            expected_cols = set(config.get('columns', []))
            actual_cols = set(df.columns)
            missing_cols = expected_cols - actual_cols
            failures = len(missing_cols)
            passed = len(missing_cols) == 0
            details = {'missing_columns': list(missing_cols)}

        elif rule.rule_type == 'uniqueness':
            col = config['column']
            duplicates = df[col].duplicated().sum()
            failures = duplicates
            passed = duplicates == 0
            details = {'duplicates': int(duplicates)}

        return DataQualityCheck.objects.create(
            rule=rule,
            dataset_path=dataset_path,
            passed=passed,
            failure_count=failures,
            details=details
        )