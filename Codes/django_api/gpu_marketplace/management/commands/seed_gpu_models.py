from django.core.management.base import BaseCommand
from gpu_marketplace.models import GPUModelSpec, PricingRule

GPU_MODELS = [
    ('RTX 3060', 'NVIDIA', 12, 'Ampere', None, None, 360, 170, 329),
    ('RTX 3070', 'NVIDIA', 8, 'Ampere', None, None, 448, 220, 499),
    ('RTX 3080', 'NVIDIA', 10, 'Ampere', None, None, 760, 320, 699),
    ('RTX 3090', 'NVIDIA', 24, 'Ampere', None, None, 936, 350, 1499),
    ('RTX 4090', 'NVIDIA', 24, 'Ada Lovelace', None, None, 1008, 450, 1599),
    ('A100-SXM4-40GB', 'NVIDIA', 40, 'Ampere', 156, 312, 1555, 400, None),
    ('A100-SXM4-80GB', 'NVIDIA', 80, 'Ampere', 156, 312, 2039, 400, None),
    ('H100-SXM-80GB', 'NVIDIA', 80, 'Hopper', 496, 990, 3350, 700, None),
    ('H200-SXM-141GB', 'NVIDIA', 141, 'Hopper', 496, 990, 4800, 700, None),
    ('B200-SXM-192GB', 'NVIDIA', 192, 'Blackwell', None, None, 8000, 1000, None),
]

DEFAULT_PRICES = {
    'RTX 3060': 0.15, 'RTX 3070': 0.20, 'RTX 3080': 0.30,
    'RTX 3090': 0.45, 'RTX 4090': 0.60,
    'A100-SXM4-40GB': 0.80, 'A100-SXM4-80GB': 1.10,
    'H100-SXM-80GB': 1.80, 'H200-SXM-141GB': 2.20, 'B200-SXM-192GB': 2.80,
}

class Command(BaseCommand):
    help = 'Seed GPU model specifications and pricing rules'

    def handle(self, *args, **options):
        for name, mfr, vram, arch, tf32, fp16, bw, tdp, msrp in GPU_MODELS:
            GPUModelSpec.objects.get_or_create(
                name=name,
                defaults={
                    'manufacturer': mfr, 'vram_gb': vram, 'architecture': arch,
                    'tf32_tflops': tf32, 'fp16_tflops': fp16,
                    'memory_bandwidth_gbs': bw, 'tdp_watts': tdp,
                    'market_price_usd': msrp,
                }
            )
        for name, price in DEFAULT_PRICES.items():
            PricingRule.objects.get_or_create(
                gpu_model=name,
                defaults={'base_price_per_hour': price}
            )
        self.stdout.write(self.style.SUCCESS(f'Seeded {len(GPU_MODELS)} GPU models'))