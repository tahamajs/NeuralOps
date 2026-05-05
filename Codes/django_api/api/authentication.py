# api/authentication.py
from rest_framework.authentication import TokenAuthentication
from rest_framework.exceptions import AuthenticationFailed

class CookieTokenAuthentication(TokenAuthentication):
    def authenticate(self, request):
        # First try the Authorization header (standard Bearer token)
        auth = request.META.get('HTTP_AUTHORIZATION', '').split()
        
        if auth and auth[0].lower() == 'bearer':
            if len(auth) == 1:
                raise AuthenticationFailed('Invalid token header. No credentials provided.')
            elif len(auth) > 2:
                raise AuthenticationFailed('Invalid token header. Token string should not contain spaces.')
            token = auth[1]
            return self.authenticate_credentials(token)
        
        # Then try the cookie as fallback
        token = request.COOKIES.get('authToken')
        if not token:
            return None
        
        try:
            return self.authenticate_credentials(token)
        except AuthenticationFailed:
            return None