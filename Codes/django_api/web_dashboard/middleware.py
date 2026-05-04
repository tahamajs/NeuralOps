from rest_framework.authtoken.models import Token
from django.contrib.auth.models import AnonymousUser
from django.shortcuts import redirect

class TokenAuthMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip login, register, static, and API endpoints
        skip_paths = ('/login', '/register', '/static', '/api/')
        if request.path.startswith(skip_paths):
            return self.get_response(request)

        # Try to get token from cookie
        token_key = request.COOKIES.get('authToken')
        if token_key:
            try:
                token = Token.objects.get(key=token_key)
                request.user = token.user
            except Token.DoesNotExist:
                return redirect(f'/login/?next={request.path}')
        else:
            return redirect(f'/login/?next={request.path}')

        return self.get_response(request)
    






from django.http import HttpResponse

class WellKnownIgnoreMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith('/.well-known/'):
            # Return 204 No Content – browser will stop retrying
            return HttpResponse(status=204)
        return self.get_response(request)