# profiles/views.py
"""
User profile management, social authentication, avatar upload, and user search.
"""

import logging

from allauth.socialaccount.models import SocialAccount, SocialApp, SocialToken
from django.conf import settings
from django.contrib.auth.models import User
from django.db.models import Q
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.authtoken.models import Token
from rest_framework.decorators import api_view, permission_classes
from rest_framework.pagination import PageNumberPagination
from rest_framework.permissions import AllowAny, IsAuthenticated
from rest_framework.response import Response

from .models import Profile
from .serializers import (
    ProfileSerializer,
    ProfileUpdateSerializer,
    UserSearchSerializer,
)

logger = logging.getLogger(__name__)


class StandardPagination(PageNumberPagination):
    page_size = 20
    page_size_query_param = "page_size"
    max_page_size = 100


# ═══════════════════════════════════════════════════════════════
# My Profile
# ═══════════════════════════════════════════════════════════════


@api_view(["GET", "PATCH"])
@permission_classes([IsAuthenticated])
def my_profile(request):
    """Retrieve or update the authenticated user's profile."""
    profile, created = Profile.objects.get_or_create(user=request.user)

    if request.method == "GET":
        serializer = ProfileSerializer(profile)
        return Response(serializer.data)

    elif request.method == "PATCH":
        serializer = ProfileUpdateSerializer(profile, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            # Return the full updated profile
            return Response(ProfileSerializer(profile).data)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


# ═══════════════════════════════════════════════════════════════
# View Any User's Profile
# ═══════════════════════════════════════════════════════════════


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_profile(request, user_id):
    """View the profile of any user by their ID."""
    profile = get_object_or_404(Profile, user_id=user_id)
    serializer = ProfileSerializer(profile)
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_profile_by_username(request, username):
    """View the profile of any user by their username."""
    profile = get_object_or_404(Profile, user__username=username)
    serializer = ProfileSerializer(profile)
    return Response(serializer.data)


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def user_profile_by_identifier(request, identifier):
    """View profile by numeric user_id or username string."""
    if str(identifier).isdigit():
        profile = Profile.objects.filter(user_id=int(identifier)).first()
    else:
        profile = Profile.objects.filter(user__username=identifier).first()
    if not profile:
        profile = get_object_or_404(Profile, user__username=identifier)
    serializer = ProfileSerializer(profile)
    return Response(serializer.data)


# ═══════════════════════════════════════════════════════════════
# User Search
# ═══════════════════════════════════════════════════════════════


@api_view(["GET"])
@permission_classes([IsAuthenticated])
def search_users(request):
    """Search users by username, display name, affiliation, or bio."""
    q = request.query_params.get("q", "").strip()
    if len(q) < 2:
        return Response({"results": [], "total": 0})

    # Search in User model fields
    user_qs = User.objects.filter(
        Q(username__icontains=q)
        | Q(first_name__icontains=q)
        | Q(last_name__icontains=q)
        | Q(email__icontains=q)
    ).distinct()[:30]

    # Also search in Profile fields
    profile_qs = Profile.objects.filter(
        Q(display_name__icontains=q) | Q(affiliation__icontains=q) | Q(bio__icontains=q)
    ).select_related("user")[:30]

    # Collect unique results
    results = []
    seen_ids = set()
    for u in user_qs:
        if u.id not in seen_ids:
            seen_ids.add(u.id)
            results.append(UserSearchSerializer(u).data)
    for p in profile_qs:
        if p.user_id not in seen_ids:
            seen_ids.add(p.user_id)
            results.append(UserSearchSerializer(p.user).data)

    return Response({"results": results, "total": len(results)})


# ═══════════════════════════════════════════════════════════════
# Avatar Management
# ═══════════════════════════════════════════════════════════════


@api_view(["POST"])
@permission_classes([IsAuthenticated])
def upload_avatar(request):
    """Upload a new avatar for the authenticated user."""
    profile, _ = Profile.objects.get_or_create(user=request.user)
    uploaded_file = request.FILES.get("avatar")
    if not uploaded_file:
        return Response({"error": "avatar file required"}, status=400)

    # Delete old avatar if exists
    if profile.avatar:
        profile.avatar.delete(save=False)

    profile.avatar = uploaded_file
    profile.save(update_fields=["avatar"])

    return Response({"avatar_url": request.build_absolute_uri(profile.avatar.url)})


@api_view(["DELETE"])
@permission_classes([IsAuthenticated])
def remove_avatar(request):
    """Remove the avatar of the authenticated user."""
    profile, _ = Profile.objects.get_or_create(user=request.user)
    if profile.avatar:
        profile.avatar.delete(save=False)
        profile.avatar = None
        profile.save(update_fields=["avatar"])
    return Response({"status": "removed"})


# ═══════════════════════════════════════════════════════════════
# Social Authentication Endpoints
# ═══════════════════════════════════════════════════════════════


@api_view(["GET"])
@permission_classes([AllowAny])
def social_login_urls(request):
    """
    Return OAuth login URLs for GitHub and HuggingFace.
    Used by the frontend to display social login buttons.
    """
    from django.urls import reverse

    providers = {}
    for provider in ["github", "huggingface"]:
        try:
            # Ensure a SocialApp exists for the provider
            app = SocialApp.objects.get(provider=provider)
            url_name = f"{provider}_login"
            providers[provider] = request.build_absolute_uri(reverse(url_name))
        except SocialApp.DoesNotExist:
            pass
    return Response(providers)


@api_view(["POST"])
@permission_classes([AllowAny])
def social_callback(request):
    """
    Exchange an OAuth authorization code for a DRF token.
    Body: { "provider": "github", "code": "...", "redirect_uri": "..." }
    """
    import requests

    provider = request.data.get("provider")
    code = request.data.get("code")
    redirect_uri = request.data.get("redirect_uri")

    if not provider or not code or not redirect_uri:
        return Response(
            {"error": "provider, code, and redirect_uri are required"}, status=400
        )

    user = None

    # ── GitHub ──────────────────────────────────────────────────
    if provider == "github":
        # Exchange code for access token
        token_resp = requests.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": getattr(settings, "GITHUB_CLIENT_ID", ""),
                "client_secret": getattr(settings, "GITHUB_CLIENT_SECRET", ""),
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
        ).json()
        access_token = token_resp.get("access_token")
        if not access_token:
            return Response(
                {"error": "Invalid GitHub code", "details": token_resp}, status=400
            )

        # Get GitHub user info
        user_resp = requests.get(
            "https://api.github.com/user",
            headers={"Authorization": f"token {access_token}"},
        ).json()
        github_id = str(user_resp.get("id"))
        username = user_resp.get("login")
        email = user_resp.get("email") or f"{username}@github.local"

        # Get or create Django user
        user, created = User.objects.get_or_create(
            username=username, defaults={"email": email}
        )

        # Store social account and token
        app = SocialApp.objects.get(provider="github")
        account, _ = SocialAccount.objects.get_or_create(
            user=user, provider="github", defaults={"uid": github_id}
        )
        SocialToken.objects.update_or_create(
            account=account, app=app, defaults={"token": access_token}
        )

    # ── HuggingFace ────────────────────────────────────────────
    elif provider == "huggingface":
        # Exchange code for token
        token_resp = requests.post(
            "https://huggingface.co/oauth/token",
            data={
                "client_id": getattr(settings, "HF_CLIENT_ID", ""),
                "client_secret": getattr(settings, "HF_CLIENT_SECRET", ""),
                "code": code,
                "grant_type": "authorization_code",
                "redirect_uri": redirect_uri,
            },
        ).json()
        access_token = token_resp.get("access_token")
        if not access_token:
            return Response(
                {"error": "Invalid HF code", "details": token_resp}, status=400
            )

        # Get HF user info
        user_resp = requests.get(
            "https://huggingface.co/oauth/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        ).json()
        hf_sub = user_resp.get("sub")
        username = user_resp.get("preferred_username") or f"hf_{hf_sub}"
        email = user_resp.get("email") or f"{username}@hf.local"

        # Get or create Django user
        user, created = User.objects.get_or_create(
            username=username, defaults={"email": email}
        )

        # Store social account and token
        try:
            app = SocialApp.objects.get(provider="huggingface")
        except SocialApp.DoesNotExist:
            # If not configured as a SocialApp, skip storing token
            pass
        else:
            account, _ = SocialAccount.objects.get_or_create(
                user=user, provider="huggingface", defaults={"uid": hf_sub}
            )
            SocialToken.objects.update_or_create(
                account=account, app=app, defaults={"token": access_token}
            )
    else:
        return Response({"error": "Unknown provider"}, status=400)

    # ── Generate DRF token ────────────────────────────────────
    if user is not None:
        token, _ = Token.objects.get_or_create(user=user)
        return Response(
            {
                "token": token.key,
                "username": user.username,
                "user_id": user.id,
                "email": user.email,
            }
        )
    else:
        return Response({"error": "Failed to authenticate user"}, status=500)
