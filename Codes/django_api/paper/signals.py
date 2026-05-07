import os
from django.db.models.signals import post_delete
from django.dispatch import receiver
from .models import PaperCompile


@receiver(post_delete, sender=PaperCompile)
def delete_compiled_pdf(sender, instance, **kwargs):
    """Delete the PDF file from storage when a compile record is deleted."""
    if instance.pdf:
        if os.path.isfile(instance.pdf.path):
            os.remove(instance.pdf.path)


@receiver(post_delete, sender=PaperCompile)
def cleanup_orphan_files(sender, instance, **kwargs):
    """Placeholder for additional cleanup."""
    pass