from allauth.socialaccount.providers.oauth2.views import (
    OAuth2Adapter, OAuth2LoginView, OAuth2CallbackView
)
from .provider import HuggingFaceProvider

class HuggingFaceOAuth2Adapter(OAuth2Adapter):
    provider_id = HuggingFaceProvider.id
    authorize_url = 'https://huggingface.co/oauth/authorize'
    access_token_url = 'https://huggingface.co/oauth/token'
    profile_url = 'https://huggingface.co/oauth/userinfo'

    def complete_login(self, request, app, token, **kwargs):
        resp = requests.get(self.profile_url, headers={
            'Authorization': f'Bearer {token.token}'
        })
        resp.raise_for_status()
        extra_data = resp.json()
        return self.get_provider().sociallogin_from_response(request, extra_data)

oauth2_login = OAuth2LoginView.adapter_view(HuggingFaceOAuth2Adapter)
oauth2_callback = OAuth2CallbackView.adapter_view(HuggingFaceOAuth2Adapter)