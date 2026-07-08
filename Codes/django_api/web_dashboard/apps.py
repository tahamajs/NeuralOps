from django.apps import AppConfig


class WebDashboardConfig(AppConfig):
    name = "web_dashboard"
    verbose_name = "Web Dashboard"

    def ready(self):
        import web_dashboard.signals  # noqa
