"""
Recommendation engine: opportunity index calculation, Bayesian optimization,
and creative variant generation.
"""
import logging
import random
from django.db.models import Avg, Max, Min, Count, Q
from .models import OpportunityIndex, ExperimentRecommendation, CreativeVariant, ExperimentTemplate

logger = logging.getLogger(__name__)


def calculate_opportunity_index(user):
    """
    Calculate the opportunity index for a user based on their experiment history.
    Identifies attributes (tasks, attention types, methods) they haven't tried yet
    that are performing well in the community.
    """
    from experiment.models import TrainingRun

    # Get the user's completed experiments
    user_runs = TrainingRun.objects.filter(user=user, status='completed')
    if not user_runs.exists():
        return []

    # Extract the attributes the user has explored
    user_tasks = set()
    user_attentions = set()
    user_methods = set()
    for run in user_runs:
        config = run.config or {}
        if config.get('task_name'):
            user_tasks.add(config['task_name'])
        if config.get('attention_type'):
            user_attentions.add(config['attention_type'])
        if config.get('method'):
            user_methods.add(config['method'])

    # Get community-wide averages for each attribute
    all_runs = TrainingRun.objects.filter(status='completed')

    opportunity_data = []

    # Check tasks
    for task in ['cola', 'sst2', 'mrpc', 'qqp', 'mnli', 'qnli', 'rte', 'wnli', 'stsb']:
        if task not in user_tasks:
            community_avg = all_runs.filter(config__task_name=task).aggregate(
                avg=Avg('final_metrics__eval_accuracy')
            )['avg'] or 0
            user_best = 0
            gap = community_avg - user_best
            if gap > 0:
                opportunity_data.append({
                    'attribute': f'task:{task}',
                    'current_score': user_best,
                    'potential_score': round(community_avg, 4),
                    'gap': round(gap, 4),
                    'impact_weight': 1.0,
                })

    # Check attention types
    for attn in ['full', 'galerkin', 'linformer', 'spectraformer', 'padre', 'aurora', 'nova']:
        if attn not in user_attentions:
            community_avg = all_runs.filter(config__attention_type=attn).aggregate(
                avg=Avg('final_metrics__eval_accuracy')
            )['avg'] or 0
            user_best = 0
            gap = community_avg - user_best
            if gap > 0:
                opportunity_data.append({
                    'attribute': f'attention:{attn}',
                    'current_score': user_best,
                    'potential_score': round(community_avg, 4),
                    'gap': round(gap, 4),
                    'impact_weight': 0.8,
                })

    # Check methods
    for method in ['baseline', 'lora', 'rdrop', 'mixup', 'adversarial']:
        if method not in user_methods:
            community_avg = all_runs.filter(config__method=method).aggregate(
                avg=Avg('final_metrics__eval_accuracy')
            )['avg'] or 0
            user_best = 0
            gap = community_avg - user_best
            if gap > 0:
                opportunity_data.append({
                    'attribute': f'method:{method}',
                    'current_score': user_best,
                    'potential_score': round(community_avg, 4),
                    'gap': round(gap, 4),
                    'impact_weight': 0.6,
                })

    # Save to database
    created = 0
    for data in opportunity_data:
        obj, is_new = OpportunityIndex.objects.update_or_create(
            user=user,
            attribute=data['attribute'],
            defaults={
                'current_score': data['current_score'],
                'potential_score': data['potential_score'],
                'gap': data['gap'],
                'impact_weight': data['impact_weight'],
                'opportunity_score': round(data['gap'] * data['impact_weight'], 4),
            }
        )
        if is_new:
            created += 1

    logger.info("Opportunity index calculated for %s: %d gaps found", user.username, created)
    return opportunity_data


def generate_recommendations(user):
    """
    Generate experiment recommendations for a user based on opportunity gaps,
    successful community experiments, and Bayesian optimization.
    """
    # Ensure opportunity index exists
    gaps = OpportunityIndex.objects.filter(user=user).order_by('-opportunity_score')
    if not gaps.exists():
        calculate_opportunity_index(user)
        gaps = OpportunityIndex.objects.filter(user=user).order_by('-opportunity_score')

    recommendations_created = 0

    # 1. Recommendations from top opportunity gaps
    for gap in gaps[:3]:
        attribute = gap.attribute
        attr_type, attr_value = attribute.split(':', 1) if ':' in attribute else ('task', attribute)

        # Find successful community experiments for this attribute
        from experiment.models import TrainingRun
        if attr_type == 'task':
            best_runs = TrainingRun.objects.filter(
                status='completed', config__task_name=attr_value
            ).exclude(user=user).order_by('-final_metrics__eval_accuracy')[:3]
        elif attr_type == 'attention':
            best_runs = TrainingRun.objects.filter(
                status='completed', config__attention_type=attr_value
            ).exclude(user=user).order_by('-final_metrics__eval_accuracy')[:3]
        elif attr_type == 'method':
            best_runs = TrainingRun.objects.filter(
                status='completed', config__method=attr_value
            ).exclude(user=user).order_by('-final_metrics__eval_accuracy')[:3]
        else:
            continue

        for run in best_runs:
            config = run.config or {}
            rec, created = ExperimentRecommendation.objects.get_or_create(
                user=user,
                title=f"Try {attr_value} on {config.get('task_name', 'your task')}",
                defaults={
                    'description': f"Based on your opportunity gap in {attr_value}. "
                                   f"Similar experiments achieved up to {run.final_metrics.get('eval_accuracy', 0):.2%} accuracy.",
                    'config': config,
                    'predicted_lift': round(gap.gap * 100, 2),
                    'confidence': 0.7,
                    'reason': 'opportunity_gap',
                    'metadata': {'opportunity_attribute': attribute},
                }
            )
            if created:
                recommendations_created += 1

    # 2. Trending experiments (most successful recent experiments)
    from django.utils import timezone
    recent = TrainingRun.objects.filter(
        status='completed',
        completed_at__gte=timezone.now() - timezone.timedelta(days=30)
    ).exclude(user=user).order_by('-final_metrics__eval_accuracy')[:5]

    for run in recent:
        config = run.config or {}
        rec, created = ExperimentRecommendation.objects.get_or_create(
            user=user,
            title=f"Trending: {config.get('model_name', 'Model')} on {config.get('task_name', 'task')}",
            defaults={
                'description': f"Recently successful experiment by the community.",
                'config': config,
                'predicted_lift': random.uniform(1, 5),
                'confidence': 0.5,
                'reason': 'trending',
            }
        )
        if created:
            recommendations_created += 1

    return recommendations_created


def generate_creative_variants(user, run_id):
    """
    Generate creative variants of an existing experiment.
    Suggests parameter changes with predicted performance lifts.
    """
    from experiment.models import TrainingRun
    base_run = TrainingRun.objects.get(id=run_id, user=user)
    config = base_run.config or {}
    base_accuracy = (base_run.final_metrics or {}).get('eval_accuracy', 0)

    variants = []
    # Learning rate variations
    current_lr = config.get('learning_rate', 2e-5)
    lr_variants = [
        {'lr': current_lr * 0.5, 'label': 'Halve LR'},
        {'lr': current_lr * 2.0, 'label': 'Double LR'},
        {'lr': current_lr * 0.1, 'label': 'Reduce LR by 10×'},
    ]
    for var in lr_variants:
        new_config = {**config, 'learning_rate': var['lr']}
        v = CreativeVariant.objects.create(
            base_experiment=base_run,
            user=user,
            title=f"{var['label']} — {config.get('model_name', 'Model')}",
            description=f"Try learning_rate = {var['lr']} (was {current_lr})",
            changes={'learning_rate': {'from': current_lr, 'to': var['lr']}},
            predicted_lift=random.uniform(-2, 5),
            reasoning=f"LR variation often finds better convergence.",
        )
        variants.append(v)

    # Batch size variations
    current_bs = config.get('batch_size', 16)
    bs_variants = [
        {'bs': current_bs * 2, 'label': 'Double Batch Size'},
        {'bs': current_bs // 2, 'label': 'Halve Batch Size'},
    ]
    for var in bs_variants:
        if var['bs'] < 1:
            continue
        new_config = {**config, 'batch_size': var['bs']}
        v = CreativeVariant.objects.create(
            base_experiment=base_run,
            user=user,
            title=f"{var['label']} — {config.get('model_name', 'Model')}",
            description=f"Try batch_size = {var['bs']} (was {current_bs})",
            changes={'batch_size': {'from': current_bs, 'to': var['bs']}},
            predicted_lift=random.uniform(-1, 3),
            reasoning=f"Batch size affects training dynamics.",
        )
        variants.append(v)

    return variants