import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from .models import Job


class PriorityPredictor:
    def __init__(self, model_path="priority_model.pkl"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return None

    def train(self):
        jobs = Job.objects.exclude(actual_priority_outcome=None).values(
            "estimated_duration",
            "gpu_requirements",
            "memory_requirements",
            "task_name",
            "actual_priority_outcome",
        )
        if len(jobs) < 50:
            return
        df = pd.DataFrame(list(jobs))
        df = pd.get_dummies(df, columns=["task_name"])
        X = df.drop("actual_priority_outcome", axis=1)
        y = df["actual_priority_outcome"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)

    def predict_priority(self, job):
        if self.model is None:
            return None
        features = {
            "estimated_duration": job.estimated_duration or 3600,
            "gpu_requirements": job.gpu_requirements,
            "memory_requirements": job.memory_requirements,
        }
        for task in Job._meta.get_field("task_name").choices:
            col = f"task_name_{task[0]}"
            features[col] = 1 if job.task_name == task[0] else 0
        df = pd.DataFrame([features])
        if hasattr(self.model, "feature_names_in_"):
            df = df.reindex(columns=self.model.feature_names_in_, fill_value=0)
        pred = self.model.predict(df)[0]
        return max(1, min(10, int(round(pred))))
