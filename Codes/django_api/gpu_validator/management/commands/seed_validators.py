from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from gpu_validator.models import ValidatorNode

User = get_user_model()


class Command(BaseCommand):
    help = "Create test validator nodes for staff users"

    def handle(self, *args, **options):
        for user in User.objects.filter(is_staff=True):
            node, created = ValidatorNode.objects.get_or_create(
                user=user,
                defaults={
                    "staked_amount_usd": 100.00,
                    "status": "active",
                },
            )
            if created:
                self.stdout.write(f"Created validator for {user.username}")
        self.stdout.write(self.style.SUCCESS("Done"))
