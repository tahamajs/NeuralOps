from .models import MemoryEntry

class SmartMemory:
    @staticmethod
    def get_context(user, project_id=None, limit=10):
        memories = MemoryEntry.objects.filter(user=user)
        if project_id:
            memories = memories.filter(project_id=project_id)
        memories = memories.order_by('-importance', '-last_accessed_at')[:limit]
        return '\n'.join(f"- {m.key}: {m.value}" for m in memories)

    @staticmethod
    def store(user, key, value, project_id=None, importance=0.5, source='manual'):
        MemoryEntry.objects.update_or_create(
            user=user,
            project_id=project_id,
            key=key,
            defaults={'value': value, 'importance': importance, 'source': source},
        )