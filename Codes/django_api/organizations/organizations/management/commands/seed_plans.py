# organizations/management/commands/seed_plans.py
from django.core.management.base import BaseCommand
from organizations.models import Plan


class Command(BaseCommand):
    help = "Seed billing plans"

    def handle(self, *args, **options):
        plans = [
            ("Free", "free", 0, 0, 10, 3, 0, 5, False, False),
            ("Pro", "pro", 2500, 25000, 25, 10, 0, 50, False, False),
            ("Team", "team", 7500, 75000, 100, 25, 5, 200, True, True),
            (
                "Enterprise",
                "enterprise",
                20000,
                200000,
                999999,
                100,
                50,
                1000,
                True,
                True,
            ),
        ]
        for name, tier, mo, yr, mem, teams, rg, stor, sso, audit in plans:
            Plan.objects.get_or_create(
                name=name,
                defaults={
                    "tier": tier,
                    "price_monthly_cents": mo,
                    "price_yearly_cents": yr,
                    "max_members": mem,
                    "max_teams": teams,
                    "max_resource_groups": rg,
                    "storage_limit_gb": stor,
                    "includes_sso": sso,
                    "includes_audit_log": audit,
                },
            )
        self.stdout.write(self.style.SUCCESS("Plans seeded"))
