# gpu_finance/models.py
class FuturesContract(models.Model):
    buyer = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='futures_bought')
    seller = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='futures_sold',
                               null=True, blank=True)
    gpu_model = models.CharField(max_length=100)
    quantity_hours = models.IntegerField()
    strike_price = models.DecimalField(max_digits=10, decimal_places=4)
    delivery_window_start = models.DateTimeField()
    delivery_window_end = models.DateTimeField()
    STATUS_CHOICES = [('open','Open'),('matched','Matched'),('settled','Settled'),('expired','Expired')]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')
    created_at = models.DateTimeField(auto_now_add=True)