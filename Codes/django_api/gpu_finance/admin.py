from django.core.management.base import BaseCommand
from gpu_finance.models import ComputeIndex, BillingPlan

INDICES = [
    ('H100-SXM-80GB', 'NEURALOPS-H100-IDX', Decimal('2.20')),
    ('A100-SXM4-80GB', 'NEURALOPS-A100-IDX', Decimal('1.20')),
    ('RTX 4090', 'NEURALOPS-RTX4090-IDX', Decimal('0.55')),
    ('RTX 3090', 'NEURALOPS-RTX3090-IDX', Decimal('0.38')),
]

PLANS = [
    ('Free', 'free', 0, 5.0, 0, 2, False),
    ('Pro', 'pro', Decimal('25.00'), 3.0, 20, 5, False),
    ('Team', 'team', Decimal('99.00'), 2.0, 100, 25, True),
    ('Enterprise', 'enterprise', Decimal('499.00'), 1.0, 1000, 100, True),
]

class Command(BaseCommand):
    help = 'Seed compute indices and billing plans'

    def handle(self, *args, **options):
        for gpu, name, price in INDICES:
            ComputeIndex.objects.get_or_create(
                gpu_model=gpu, index_name=name,
                defaults={'current_value': price}
            )
        for name, tier, fee, markup, hrs, jobs, support in PLANS:
            BillingPlan.objects.get_or_create(
                name=name,
                defaults={'tier': tier, 'monthly_base_fee_usd': fee,
                          'gpu_markup_pct': markup, 'included_gpu_hours': hrs,
                          'max_concurrent_jobs': jobs, 'priority_support': support}
            )
        self.stdout.write(self.style.SUCCESS('Seeded indices and plans'))