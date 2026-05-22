from django.core.management.base import BaseCommand
from reputation.models import ReputationWeight


class Command(BaseCommand):
    help = 'Seed default reputation weights'

    def handle(self, *args, **options):
        weights, created = ReputationWeight.objects.get_or_create(
            pk=1,
            defaults={
                'OTHER_READS': 5.0,
                'FULL_TEXT_READS': 15.0,
                'RECOMMENDATIONS': 25.0,
                'CITATIONS': 50.0,
            }
        )
        if created:
            self.stdout.write(self.style.SUCCESS('Created default reputation weights'))
        else:
            self.stdout.write('Reputation weights already exist')