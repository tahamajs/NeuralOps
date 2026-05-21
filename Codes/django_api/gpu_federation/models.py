# gpu_federation/models.py
class FederationCluster(models.Model):
    name = models.CharField(max_length=200)
    owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    min_nodes = models.IntegerField(default=2)
    max_nodes = models.IntegerField(default=50)
    created_at = models.DateTimeField(auto_now_add=True)

class FederationNode(models.Model):
    cluster = models.ForeignKey(FederationCluster, on_delete=models.CASCADE, related_name='nodes')
    listing = models.ForeignKey('gpu_marketplace.GPUListing', on_delete=models.CASCADE)
    status = models.CharField(max_length=20, default='pending')  # pending, active, disconnected

class FederationTask(models.Model):
    cluster = models.ForeignKey(FederationCluster, on_delete=models.CASCADE, related_name='tasks')
    training_run = models.ForeignKey('jobscheduler.Job', on_delete=models.CASCADE)  # link to existing run
    shards_count = models.IntegerField(default=4)
    progress = models.FloatField(default=0)