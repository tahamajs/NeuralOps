# jobscheduler/predictive.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from .models import Job

class PriorityPredictor:
    def __init__(self):
        self.model = None
        self.features = ['task_duration_seconds', 'gpu_count', 'memory_gb', 'priority_initial']

    def train(self):
        # Load historical data (e.g., from Job model)
        df = pd.DataFrame(list(Job.objects.values(*self.features)))
        if len(df) < 50:
            return  # Not enough data to train
        
        X = df[self.features]
        y = df['final_priority_optimal']  # Target: actual priority that led to best outcome
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Save model
        joblib.dump(self.model, 'priority_model.pkl')
        return self.model

    def predict_priority(self, job):
        if self.model is None:
            self.model = joblib.load('priority_model.pkl')
        
        # Prepare features for the job (estimated duration, resource usage, etc.)
        features = pd.DataFrame([{
            'task_duration_seconds': job.estimated_duration or 3600,  # fallback
            'gpu_count': job.gpu_requirements,
            'memory_gb': job.memory_requirements,
            'priority_initial': job.priority
        }])
        predicted = self.model.predict(features)[0]
        # Map to integer priority range 1-10
        return max(1, min(10, int(round(predicted))))