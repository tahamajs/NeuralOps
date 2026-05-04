# notes/permissions.py
from rest_framework import permissions

class IsOwnerOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method in permissions.SAFE_METHODS:
            return True
        return obj.author == request.user

class IsResearcherOrReadOnly(permissions.BasePermission):
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user.groups.filter(name='Researcher').exists() or request.user.is_staff

class CanModifyPublicNote(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if obj.is_public:
            return request.user.has_perm('notes.change_researchernote') or request.user.is_staff
        return obj.author == request.user