"""
Auto‑trigger failure analysis when a training run fails.
"""
from django.db.models.signals import post_save
from django.dispatch import receiver
from .forensic_engine import FailureForensicEngine
from .models import FailureAnalysis, FixAction


@receiver(post_save, sender='experiment.TrainingRun')
def auto_analyze_on_failure(sender, instance, **kwargs):
    """When a run fails, automatically run the forensic engine."""
    if instance.status == 'failed' and not hasattr(instance, 'failure_analysis'):
        engine = FailureForensicEngine(instance)
        result = engine.analyze()

        analysis = FailureAnalysis.objects.create(
            run=instance,
            status='completed',
            root_cause_category=result['root_cause_category'],
            root_cause_detail=result['root_cause_detail'],
            confidence=result['confidence'],
            error_logs=result['error_logs'],
            anomaly_patterns=result['anomaly_patterns'],
            suggestions=result['suggestions'],
        )

        for fix_data in result['fix_actions']:
            FixAction.objects.create(
                analysis=analysis,
                action_type=fix_data['action_type'],
                description=fix_data['description'],
                config_changes=fix_data.get('config_changes', {}),
            )