# jobscheduler/management/commands/retrain_priority_model.py
from django.core.management.base import BaseCommand
from jobscheduler.predictive import PriorityPredictor

class Command(BaseCommand):
    help = 'Retrain the priority prediction model using historical job data'

    def handle(self, *args, **options):
        predictor = PriorityPredictor()
        predictor.train()
        self.stdout.write(self.style.SUCCESS('Model retrained successfully'))