import time
from celery import shared_task
from django.db import transaction
from experiment.models import TrainingRun
from .models import Sweep, SweepTrial
from .engine import suggest_params, report_metric

@shared_task(bind=True, max_retries=3)
def start_sweep(self, sweep_id):
    sweep = Sweep.objects.get(id=sweep_id)
    max_trials = sweep.max_trials
    max_failures = sweep.max_failures
    parallel_workers = sweep.parallel_workers
    consecutive_failures = 0

    while True:
        sweep.refresh_from_db()
        if sweep.status != 'active':
            break
        if sweep.trials.count() >= max_trials:
            sweep.status = 'completed'
            sweep.save(update_fields=['status'])
            break
        if consecutive_failures >= max_failures:
            sweep.status = 'failed'
            sweep.save(update_fields=['status'])
            break
        while sweep.trials.filter(status='running').count() >= parallel_workers:
            time.sleep(5)

        params, trial = suggest_params(sweep)
        trial_record = SweepTrial.objects.create(
            sweep=sweep, trial_number=trial.number,
            params=params, status='queued',
        )
        base_config = sweep.config.get('base_config', {})
        run_config = {**base_config, **params}
        run = TrainingRun.objects.create(
            experiment=sweep.project, job_type='training',
            config=run_config, status='queued',
            name=f"{sweep.name}_trial_{trial.number}"
        )
        trial_record.run = run
        trial_record.status = 'running'
        trial_record.save(update_fields=['run', 'status'])

        from experiment.tasks import train_job
        train_job.apply_async(
            args=[run.id, sweep.metric_name],
            link=report_sweep_result.s(
                sweep_id=str(sweep.id), trial_number=trial.number,
                metric_names=sweep.metric_names,
            )
        )
        time.sleep(2)

    if sweep.status == 'active':
        sweep.status = 'completed'
        sweep.save(update_fields=['status'])

@shared_task
def report_sweep_result(run_id, metric_name, sweep_id, trial_number, metric_names=None):
    try:
        run = TrainingRun.objects.get(id=run_id)
        sweep = Sweep.objects.get(id=sweep_id)
        if metric_names:
            objectives = []
            for i, mn in enumerate(metric_names):
                val = run.final_metrics.get(mn)
                objectives.append(
                    val if val is not None
                    else (float('inf') if sweep.directions[i] == 'minimize' else -float('inf'))
                )
            objective_value = objectives[0] if objectives else None
            report_metric(sweep, trial_number, objective_value=objective_value, objectives=objectives)
        else:
            objective_value = run.final_metrics.get(metric_name)
            report_metric(sweep, trial_number, objective_value=objective_value)

        trial = SweepTrial.objects.get(sweep=sweep, trial_number=trial_number)
        trial.objective_value = objective_value if not metric_names else None
        trial.objectives = objectives if metric_names else None
        trial.status = 'completed' if run.status == 'completed' else 'failed'
        trial.save()
    except Exception as e:
        print(f"Error reporting sweep result: {e}")