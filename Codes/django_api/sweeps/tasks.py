import time
from celery import shared_task
from django.db import transaction
from experiment.models import Sweep, SweepTrial, TrainingRun
from .engine import suggest_params, report_metric, get_study


@shared_task
def start_sweep(sweep_id):
    """Main loop that runs a sweep until max_trials or pause."""
    sweep = Sweep.objects.get(id=sweep_id)
    study = get_study(sweep)
    max_trials = sweep.config.get('max_trials', 20)

    while True:
        sweep.refresh_from_db()
        if sweep.status != 'active':
            break
        if sweep.trials.count() >= max_trials:
            sweep.status = 'completed'
            sweep.save(update_fields=['status'])
            break

        # Get new parameters from Optuna
        params, trial = suggest_params(sweep)

        # Save as SweepTrial
        trial_record = SweepTrial.objects.create(
            sweep=sweep,
            trial_number=trial.number,
            params=params,
            status='queued',
        )

        # Create a TrainingRun (experiment)
        base_config = sweep.config.get('base_config', {})
        run_config = {**base_config, **params}
        run = TrainingRun.objects.create(
            experiment=sweep.project,
            job_type='training',
            config=run_config,
            status='queued',
            name=f"{sweep.name}_trial_{trial.number}"
        )

        # Link trial to run
        trial_record.run = run
        trial_record.save(update_fields=['run'])

        # Submit the training job (Celery task in experiment.tasks)
        from experiment.tasks import train_job
        train_job.apply_async(
            args=[run.id, sweep.metric_name],
            link=report_sweep_result.s(
                sweep_id=str(sweep.id),
                trial_number=trial.number
            )
        )

        # Respect parallel_workers limit
        while sweep.trials.filter(status='running').count() >= sweep.config.get('parallel_workers', 1):
            time.sleep(5)

    # After loop, mark sweep as completed
    sweep.status = 'completed'
    sweep.save(update_fields=['status'])


@shared_task
def report_sweep_result(run_id, metric_name, sweep_id, trial_number):
    """Callback called when a training job finishes."""
    try:
        run = TrainingRun.objects.get(id=run_id)
        objective_value = run.final_metrics.get(metric_name, 0)
        sweep = Sweep.objects.get(id=sweep_id)
        report_metric(sweep, trial_number, objective_value)

        trial = SweepTrial.objects.get(sweep=sweep, trial_number=trial_number)
        trial.objective_value = objective_value
        trial.status = 'completed' if run.status == 'completed' else 'failed'
        trial.save()
    except Exception as e:
        print(f"Error reporting sweep result: {e}")
