# gpu_validator/models.py
class ValidatorNode(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
                             related_name='validator_nodes')
    staked_amount_usd = models.DecimalField(max_digits=10, decimal_places=2)
    total_validations = models.IntegerField(default=0)
    accuracy_score = models.FloatField(default=100.0)
    is_active = models.BooleanField(default=True)

class VerificationTask(models.Model):
    job = models.ForeignKey('gpu_marketplace.ComputeJob', on_delete=models.CASCADE)
    assigned_validators = models.ManyToManyField(ValidatorNode)
    consensus_result = models.CharField(max_length=50, choices=[('valid','Valid'),('invalid','Invalid')],
                                        null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)

class ProofOfCompute(models.Model):
    job = models.OneToOneField('gpu_marketplace.ComputeJob', on_delete=models.CASCADE,
                               related_name='proof')
    hash_signature = models.CharField(max_length=256)
    tee_attestation = models.TextField(blank=True)   # For TEE-enabled GPUs
    validator_signatures = models.JSONField(default=dict)
    verified_at = models.DateTimeField(auto_now_add=True)