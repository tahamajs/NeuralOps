# jobscheduler/tasks.py
from celery import shared_task
from .predictive import PriorityPredictor

@shared_task
def retrain_priority_model():
    """Periodic task to retrain the priority prediction model."""
    predictor = PriorityPredictor()
    predictor.train()
    return {"status": "retrained"}

# To schedule this task every 24 hours, add to celery beat schedule:
# from celery.schedules import crontab
# app.conf.beat_schedule = {
#     'retrain-priority-model': {
#         'task': 'jobscheduler.tasks.retrain_priority_model',
#         'schedule': crontab(hour=2, minute=0),  # every day at 2 AM
#     },
# }