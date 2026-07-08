from django.core.management.base import BaseCommand
from jobscheduler.models import Job
from model_versioning.utils import cleanup_old_versions


class Command(BaseCommand):
    help = "Cleanup old model versions across all runs"

    def add_arguments(self, parser):
        parser.add_argument("--keep-best", type=int, default=3)
        parser.add_argument("--keep-tags", nargs="+", default=[])

    def handle(self, *args, **options):
        for job in Job.objects.all():
            deleted = cleanup_old_versions(
                job,
                keep_best_k=options["keep_best"],
                keep_version_tags=options["keep_tags"],
            )
            self.stdout.write(f"Job {job.id}: deleted {deleted} versions")
