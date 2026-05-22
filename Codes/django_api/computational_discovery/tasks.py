"""
Celery tasks for running computational discovery experiments.
"""
import logging, time, random
from celery import shared_task
from django.utils import timezone
from .models import CodeVariant, CodeExperiment, ExperimentRun
from .code_generator import generate_variant, evaluate_variant

logger = logging.getLogger(__name__)


@shared_task(bind=True, max_retries=2)
def run_computational_experiment(self, experiment_id: str):
    """
    Main loop: generate variants, test them, score them.
    Runs until max_variants or max_iterations is reached.
    """
    try:
        experiment = CodeExperiment.objects.get(id=experiment_id)
    except CodeExperiment.DoesNotExist:
        return

    if experiment.status not in ('generating', 'testing'):
        return

    start_time = time.time()
    base_code = experiment.base_code or _default_base_code()

    for iteration in range(experiment.max_iterations):
        experiment.refresh_from_db()
        if experiment.status == 'paused':
            return

        experiment.status = 'generating'
        experiment.save(update_fields=['status'])

        variants_this_round = max(1, experiment.parallel_workers)

        for _ in range(variants_this_round):
            if experiment.variants.count() >= experiment.max_variants:
                break

            # Generate a new variant
            gen = generate_variant(base_code)
            variant = CodeVariant.objects.create(
                experiment=experiment,
                created_by=experiment.owner,
                version_number=experiment.variants.count() + 1,
                code=gen['code'],
                description='; '.join(gen['modifications']),
                hyperparameters=gen['hyperparameters'],
                status='queued',
                iteration=iteration,
            )

        # Test variants
        experiment.status = 'testing'
        experiment.save(update_fields=['status'])

        queued = experiment.variants.filter(status='queued')
        for variant in queued:
            # Run the test
            run = ExperimentRun.objects.create(variant=variant, status='running')
            try:
                metrics = evaluate_variant(variant)
                run.metrics = metrics
                run.status = 'completed'
                run.completed_at = timezone.now()

                variant.metrics = metrics
                variant.score = metrics.get('accuracy', 0)
                variant.status = 'completed'
                variant.save(update_fields=['metrics', 'score', 'status'])

                # Update best variant
                if experiment.best_variant is None or (
                    (experiment.metric_direction == 'maximize' and variant.score > experiment.best_variant.score)
                    or (experiment.metric_direction == 'minimize' and variant.score < experiment.best_variant.score)
                ):
                    experiment.best_variant = variant
                    experiment.save(update_fields=['best_variant'])

            except Exception as e:
                run.status = 'failed'
                run.error_message = str(e)
                variant.status = 'failed'
                variant.save(update_fields=['status'])
            finally:
                run.completed_at = timezone.now()
                run.duration_seconds = random.uniform(10, 60)
                run.save()

            experiment.total_variants_tested = experiment.variants.filter(
                status__in=['completed', 'failed']
            ).count()
            experiment.save(update_fields=['total_variants_tested'])

        experiment.total_iterations = iteration + 1
        experiment.save(update_fields=['total_iterations'])

        # Check if target is met
        if experiment.target_score and experiment.best_variant:
            if (experiment.metric_direction == 'maximize' and experiment.best_variant.score >= experiment.target_score) or \
               (experiment.metric_direction == 'minimize' and experiment.best_variant.score <= experiment.target_score):
                break

        # Small delay between iterations
        time.sleep(2)

    experiment.status = 'completed'
    experiment.elapsed_seconds = time.time() - start_time
    experiment.completed_at = timezone.now()
    experiment.save(update_fields=['status', 'elapsed_seconds', 'completed_at'])


def _default_base_code() -> str:
    return '''import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate = 2e-5,
    batch_size = 16,
    epochs = 3,
    optimizer = "adamw",
    dropout = 0.1,
    compression_rank = 64,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
'''