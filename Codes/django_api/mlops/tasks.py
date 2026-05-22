from celery import shared_task
from .models import *
from .inference_deployer import deploy_to_huggingface
from .pipeline_engine import PipelineExecutor
from .data_quality import DataQualityRunner
from .monitoring import run_monitors_for_endpoint

@shared_task(bind=True, max_retries=3)
def deploy_inference_endpoint(self, endpoint_id: str):
    endpoint = InferenceEndpoint.objects.get(id=endpoint_id)
    endpoint.status = 'deploying'
    endpoint.save()
    try:
        url = deploy_to_huggingface(endpoint.model_version, endpoint.config)
        endpoint.endpoint_url = url
        endpoint.status = 'running'
        endpoint.save()
    except Exception as e:
        endpoint.status = 'failed'
        endpoint.save()
        raise self.retry(exc=e)

@shared_task(bind=True, max_retries=2)
def execute_pipeline(self, run_id: str):
    run = PipelineRun.objects.get(id=run_id)
    executor = PipelineExecutor(run)
    executor.execute()

@shared_task
def run_data_quality_checks():
    for rule in DataQualityRule.objects.filter(is_active=True):
        DataQualityRunner.run_check(rule, rule.config.get('dataset_path', ''))

@shared_task(bind=True, max_retries=2)
def run_ctcd_pipeline(self, run_id: str):
    run = CTCDRun.objects.get(id=run_id)
    # 1. Train
    run.status = 'training'
    run.save()
    from experiment.models import Experiment, TrainingRun
    exp = Experiment.objects.create(name=f"CTCD: {run.pipeline.name}")
    training_run = TrainingRun.objects.create(experiment=exp, config=run.pipeline.training_config, status='queued')
    from experiment.tasks import run_training_task
    run_training_task.delay(str(training_run.id))
    # 2. Evaluate
    # 3. Deploy
    # (simplified)

@shared_task(bind=True, max_retries=2)
def run_distillation(self, job_id: str):
    job = DistillationJob.objects.get(id=job_id)
    # Execute distillation using framework
    from framework.distillation import distill_model
    output = distill_model(job.teacher_model.artifact_path, job.student_model_config, job.compression_config)
    # Save output model version
    from model_versioning.services import create_version
    version = create_version(run=job.teacher_model.run, metrics=output['metrics'], artifact_dir=output['path'])
    job.output_model = version
    job.status = 'completed'
    job.save()

@shared_task(bind=True, max_retries=2)
def launch_multi_node_training(self, training_id: str):
    training = MultiNodeTraining.objects.get(id=training_id)
    # Use federation cluster to distribute training
    from gpu_federation.coordinator import FederationCoordinator
    # Launch across nodes

@shared_task
def monitor_all_endpoints():
    for endpoint in InferenceEndpoint.objects.filter(status='running'):
        run_monitors_for_endpoint(endpoint)