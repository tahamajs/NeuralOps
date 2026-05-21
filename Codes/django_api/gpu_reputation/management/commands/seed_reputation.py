from django.core.management.base import BaseCommand
from gpu_reputation.models import SlashingCondition, ReputationBadge

class Command(BaseCommand):
    help = 'Seed reputation conditions and badges'

    def handle(self, *args, **options):
        # Slashing conditions (EigenLayer-style)
        conditions = [
            ('downtime_1h', 'Downtime > 1 Hour', 'downtime', 0.0001, False, 'burn'),
            ('job_failure', 'Failed Compute Job', 'job_failure', 0.05, True, 'victim'),
            ('fraud', 'Fraudulent Output', 'fraud', 0.50, True, 'victim'),
            ('dispute_lost', 'Dispute Lost', 'dispute_lost', 0.10, True, 'platform'),
            ('sla_violation', 'SLA Violation', 'sla_violation', 0.20, True, 'victim'),
        ]
        for name, desc, ctype, frac, redist, recip in conditions:
            SlashingCondition.objects.get_or_create(
                name=name,
                defaults={'description': desc, 'condition_type': ctype,
                          'slash_fraction': frac, 'redistribution': redist,
                          'redistribution_recipient': recip}
            )

        # Reputation badges
        badges = [
            ('First Job', 'Complete your first compute job', 'bronze', '🥉', 0, None, 1),
            ('Reliable Provider', 'Provider score > 80', 'silver', '🥈', 80, None, 10),
            ('Top Provider', 'Provider score > 95', 'gold', '🥇', 95, None, 50),
            ('Trusted Consumer', 'Consumer score > 80', 'silver', '🥈', None, 80, 5),
            ('Clean Record', 'No disputes lost', 'gold', '🏆', None, None, 20, 0),
            ('Centurion', '100+ jobs completed', 'platinum', '💎', None, None, 100),
        ]
        for name, desc, tier, icon, prov, cons, jobs, *disputes in badges:
            max_disputes = disputes[0] if disputes else None
            ReputationBadge.objects.get_or_create(
                name=name,
                defaults={'description': desc, 'tier': tier, 'icon': icon,
                          'min_provider_score': prov, 'min_consumer_score': cons,
                          'min_jobs_completed': jobs, 'max_disputes_lost': max_disputes}
            )
        self.stdout.write(self.style.SUCCESS('Seeded conditions and badges'))