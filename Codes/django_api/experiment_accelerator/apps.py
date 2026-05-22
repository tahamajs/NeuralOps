from django.apps import AppConfig

class ExperimentAcceleratorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'experiment_accelerator'

    def ready(self):
        import experiment_accelerator.signals  # noqa