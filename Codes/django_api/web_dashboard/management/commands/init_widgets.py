from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from web_dashboard.models import DEFAULT_WIDGETS, DashboardWidget


class Command(BaseCommand):
    help = "Initialize default dashboard widgets for all users"

    def handle(self, *args, **kwargs):
        User = get_user_model()
        for user in User.objects.all():
            if not DashboardWidget.objects.filter(user=user).exists():
                for w in DEFAULT_WIDGETS:
                    DashboardWidget.objects.create(user=user, **w)
                self.stdout.write(f"Created default widgets for {user.username}")
