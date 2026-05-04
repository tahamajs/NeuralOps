# web_dashboard/middleware.py
from django.shortcuts import redirect
from rest_framework.authtoken.models import Token
from django.contrib.auth.models import AnonymousUser

class TokenAuthMiddleware:
    """Authenticate user from authToken cookie (for page views, not API)."""
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip authentication for login, register, static, API endpoints
        skip_paths = ('/login/', '/register/', '/static/', '/api/', '/.well-known/')
        if request.path.startswith(skip_paths):
            return self.get_response(request)

        token_key = request.COOKIES.get('authToken')
        if token_key:
            try:
                token = Token.objects.get(key=token_key)
                request.user = token.user
            except Token.DoesNotExist:
                # Not authenticated, redirect to login
                return redirect(f'/login/?next={request.path}')
        else:
            return redirect(f'/login/?next={request.path}')

        return self.get_response(request)

class WellKnownIgnoreMiddleware:
    """Return 204 for .well-known requests (e.g., browser devtools)."""
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith('/.well-known/'):
            from django.http import HttpResponse
            return HttpResponse(status=204)
        return self.get_response(request)