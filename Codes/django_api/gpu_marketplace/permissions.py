from rest_framework import permissions

class IsProviderOrReadOnly(permissions.BasePermission):
    """Only the listing provider can edit; everyone can read."""
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.provider == request.user

class IsJobParticipant(permissions.BasePermission):
    """Only the consumer or provider of a job can view it."""
    def has_object_permission(self, request, view, obj):
        return obj.consumer == request.user or obj.provider == request.user