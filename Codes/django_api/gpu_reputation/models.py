# gpu_reputation/models.py
class ReputationScore(models.Model):
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    provider_score = models.FloatField(default=5.0)  # as a provider
    consumer_score = models.FloatField(default=5.0)  # as a consumer
    total_jobs_completed = models.IntegerField(default=0)
    total_disputes_lost = models.IntegerField(default=0)

class Stake(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    amount_usd = models.DecimalField(max_digits=10, decimal_places=2)
    locked_until = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

class Dispute(models.Model):
    job = models.ForeignKey('gpu_marketplace.ComputeJob', on_delete=models.CASCADE)
    raised_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    reason = models.TextField()
    resolution = models.CharField(max_length=50, null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

class SlashingEvent(models.Model):
    stake = models.ForeignKey(Stake, on_delete=models.CASCADE)
    amount_slashed = models.DecimalField(max_digits=10, decimal_places=2)
    reason = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)