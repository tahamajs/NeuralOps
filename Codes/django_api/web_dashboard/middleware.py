from django.shortcuts import redirect
from rest_framework.authtoken.models import Token


class TokenAuthMiddleware:
    """Authenticate user from authToken cookie for page views."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        # Skip authentication for these paths
        skip_prefixes = ("/login/", "/register/", "/static/", "/api/", "/.well-known/")
        if request.path.startswith(skip_prefixes):
            return self.get_response(request)

        if getattr(request, "user", None) and request.user.is_authenticated:
            return self.get_response(request)

        token_key = request.COOKIES.get("authToken")
        if not token_key:
            auth_header = request.headers.get("Authorization") or request.META.get("HTTP_AUTHORIZATION")
            if auth_header and auth_header.startswith("Token "):
                token_key = auth_header.split(" ")[1].strip()

        if token_key:
            try:
                token = Token.objects.get(key=token_key)
                request.user = token.user
            except Token.DoesNotExist:
                return redirect(f"/login/?next={request.path}")
        else:
            return redirect(f"/login/?next={request.path}")

        return self.get_response(request)


class WellKnownIgnoreMiddleware:
    """Return 204 for .well-known requests."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        if request.path.startswith("/.well-known/"):
            from django.http import HttpResponse

            return HttpResponse(status=204)
        return self.get_response(request)
