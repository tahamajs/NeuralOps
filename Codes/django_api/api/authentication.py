# api/authentication.py
from rest_framework.authentication import TokenAuthentication

class CookieTokenAuthentication(TokenAuthentication):
    def authenticate(self, request):
        # First try the Authorization header
        auth = super().authenticate(request)
        if auth is not None:
            return auth
        # Then try the cookie
        token = request.COOKIES.get('authToken')
        if not token:
            return None
        return self.authenticate_credentials(token)