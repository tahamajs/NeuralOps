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
    







# jobscheduler/predictive.py
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from .models import Job

class PriorityPredictor:
    def __init__(self, model_path='priority_model.pkl'):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        try:
            return joblib.load(self.model_path)
        except FileNotFoundError:
            return None

    def train(self):
        # Gather historical data (assuming we have a column 'actual_priority_outcome')
        df = pd.DataFrame(list(Job.objects.values(
            'estimated_duration', 'gpu_requirements', 'memory_requirements',
            'task_name', 'actual_priority_outcome'
        )))
        if len(df) < 50:
            return  # Not enough data

        # One‑hot encode task_name
        df = pd.get_dummies(df, columns=['task_name'])
        X = df.drop('actual_priority_outcome', axis=1)
        y = df['actual_priority_outcome']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        joblib.dump(self.model, self.model_path)
        return self.model

    def predict_priority(self, job):
        if self.model is None:
            return None
        # Build feature vector
        features = {
            'estimated_duration': job.estimated_duration or 3600,
            'gpu_requirements': job.gpu_requirements,
            'memory_requirements': job.memory_requirements or 8,
        }
        # Add one‑hot for task name
        for task in Job._meta.get_field('task_name').choices:
            col = f'task_name_{task[0]}'
            features[col] = 1 if job.task_name == task[0] else 0
        df = pd.DataFrame([features])
        pred = self.model.predict(df)[0]
        # Clip to 1‑10 range
        return max(1, min(10, int(round(pred))))