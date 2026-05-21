from django.contrib import admin
from .models import *

@admin.register(BenchmarkSuite)
class BenchmarkSuiteAdmin(admin.ModelAdmin):
    list_display = ('name', 'suite_type', 'version', 'is_active', 'created_at')
    list_filter = ('suite_type', 'is_active')

@admin.register(BenchmarkTest)
class BenchmarkTestAdmin(admin.ModelAdmin):
    list_display = ('name', 'suite', 'category', 'metric_name', 'weight', 'is_active')
    list_filter = ('category', 'suite', 'is_active')

@admin.register(BenchmarkRun)
class BenchmarkRunAdmin(admin.ModelAdmin):
    list_display = ('id_short', 'listing', 'suite', 'status', 'created_at')
    list_filter = ('status',)

    def id_short(self, obj): return obj.id.hex[:8]

@admin.register(BenchmarkResult)
class BenchmarkResultAdmin(admin.ModelAdmin):
    list_display = ('test', 'listing', 'batch_size', 'precision', 'metric_value', 'normalized_score')
    list_filter = ('test__suite', 'precision', 'is_valid')

@admin.register(ListingBenchmarkScore)
class ListingBenchmarkScoreAdmin(admin.ModelAdmin):
    list_display = ('listing', 'composite_score', 'training_score', 'inference_score', 'llm_score')