"""Celery tasks for the Lab Assistant."""

from celery import shared_task

from .ai_assistant import AIAssistant


@shared_task
def auto_pre_label_new_tasks(project_id: str):
    """Automatically run AI pre-labeling on newly imported tasks."""
    from .models import AnnotationProject

    try:
        project = AnnotationProject.objects.get(id=project_id)
    except AnnotationProject.DoesNotExist:
        return

    if not project.ai_model_id:
        return

    assistant = AIAssistant(model_id=project.ai_model_id)
    tasks = project.tasks.filter(status="pending", ai_predictions=[])
    count = 0
    for task in tasks:
        if assistant.pre_label_task(task):
            count += 1
    return count
