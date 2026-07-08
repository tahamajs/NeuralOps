from allauth.socialaccount.providers.base import ProviderAccount
from allauth.socialaccount.providers.oauth2.provider import OAuth2Provider


class HuggingFaceAccount(ProviderAccount):
    def get_profile_url(self):
        return f"https://huggingface.co/{self.account.extra_data.get('preferred_username')}"

    def get_avatar_url(self):
        return self.account.extra_data.get("avatarUrl")

    def to_str(self):
        return self.account.extra_data.get("preferred_username", "")


class HuggingFaceProvider(OAuth2Provider):
    id = "huggingface"
    name = "HuggingFace"
    account_class = HuggingFaceAccount

    def extract_uid(self, data):
        return str(data["sub"])

    def extract_common_fields(self, data):
        return {
            "username": data.get("preferred_username"),
            "email": data.get("email"),
            "first_name": (
                data.get("name", "").split(" ")[0] if data.get("name") else ""
            ),
            "last_name": (
                " ".join(data.get("name", "").split(" ")[1:])
                if data.get("name")
                else ""
            ),
        }

    def get_default_scope(self):
        return ["openid", "profile", "email"]


provider_classes = [HuggingFaceProvider]
