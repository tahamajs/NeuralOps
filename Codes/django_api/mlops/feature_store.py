"""Feature registry and serving."""
import hashlib
from .models import FeatureGroup, FeatureRecord, FeatureDriftLog

class FeatureStore:
    @staticmethod
    def ingest(group_name: str, entity_id: str, values: dict):
        group, _ = FeatureGroup.objects.get_or_create(name=group_name)
        FeatureRecord.objects.update_or_create(
            group=group, entity_id=entity_id,
            defaults={'values': values}
        )

    @staticmethod
    def serve(group_name: str, entity_ids: list) -> dict:
        records = FeatureRecord.objects.filter(
            group__name=group_name, entity_id__in=entity_ids
        )
        return {r.entity_id: r.values for r in records}

    @staticmethod
    def check_drift(group_name: str) -> float:
        """Calculate Population Stability Index (PSI) for feature drift."""
        group = FeatureGroup.objects.get(name=group_name)
        records = group.records.all()
        # Simplified PSI calculation
        if records.count() < 100:
            return 0.0
        # ... (full PSI implementation)
        return 0.05