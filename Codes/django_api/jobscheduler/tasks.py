# tasks.py (Celery)
from celery import shared_task
from jobscheduler.predictive import PriorityPredictor

@shared_task
def retrain_priority_model():
    predictor = PriorityPredictor()
    predictor.train()