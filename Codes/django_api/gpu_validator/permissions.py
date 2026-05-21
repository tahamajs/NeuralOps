from rest_framework import permissions

class IsActiveValidator(permissions.BasePermission):
    """Only active validators can perform verification actions."""
    def has_permission(self, request, view):
        return (
            request.user.is_authenticated
            and hasattr(request.user, 'validator_node')
            and request.user.validator_node.status == 'active'
        )