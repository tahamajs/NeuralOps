@api_view(['GET'])
@permission_classes([AllowAny])
def social_login_urls(request):
    """Return OAuth login URLs for GitHub and HuggingFace."""
    from allauth.socialaccount.models import SocialApp
    from django.urls import reverse
    providers = {}
    for provider in ['github', 'huggingface']:
        try:
            app = SocialApp.objects.get(provider=provider)
            url = reverse(f'{provider}_login')
            providers[provider] = request.build_absolute_uri(url)
        except SocialApp.DoesNotExist:
            pass
    return Response(providers)


@api_view(['POST'])
@permission_classes([AllowAny])
def social_callback(request):
    """
    Exchange OAuth authorization code for a DRF token.
    Body: { "provider": "github", "code": "...", "redirect_uri": "..." }
    """
    import requests
    provider = request.data.get('provider')
    code = request.data.get('code')
    redirect_uri = request.data.get('redirect_uri')

    if provider == 'github':
        # Exchange code for access token
        token_resp = requests.post(
            'https://github.com/login/oauth/access_token',
            data={
                'client_id': settings.GITHUB_CLIENT_ID,
                'client_secret': settings.GITHUB_CLIENT_SECRET,
                'code': code,
                'redirect_uri': redirect_uri,
            },
            headers={'Accept': 'application/json'},
        ).json()
        access_token = token_resp.get('access_token')
        if not access_token:
            return Response({'error': 'Invalid code'}, status=400)
        # Get user info
        user_resp = requests.get('https://api.github.com/user', headers={
            'Authorization': f'token {access_token}'
        }).json()
        username = user_resp.get('login')
        email = user_resp.get('email')
        # Find or create user
        user, _ = User.objects.get_or_create(username=username, defaults={'email': email})
        # Create/update social token
        from allauth.socialaccount.models import SocialApp, SocialAccount, SocialToken
        app = SocialApp.objects.get(provider='github')
        account, _ = SocialAccount.objects.get_or_create(user=user, provider='github', defaults={'uid': str(user_resp['id'])})
        SocialToken.objects.update_or_create(account=account, app=app, defaults={'token': access_token})
    elif provider == 'huggingface':
        # similar flow using HF OAuth
        pass
    else:
        return Response({'error': 'Unknown provider'}, status=400)

    # Generate DRF token
    from rest_framework.authtoken.models import Token
    token, _ = Token.objects.get_or_create(user=user)
    return Response({
        'token': token.key,
        'username': user.username,
        'user_id': user.id,
    })


# ── GitHub Integration ─────────────────────────────────────────
import base64, json
from github import Github, GithubException

def _get_github_client(user):
    """Return a PyGithub client authenticated with the user's GitHub token."""
    from allauth.socialaccount.models import SocialToken
    token = SocialToken.objects.filter(account__user=user, app__provider='github').first()
    if not token:
        raise PermissionError("GitHub not connected. Please log in with GitHub.")
    return Github(token.token)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def github_repos(request):
    """List the user's GitHub repositories."""
    try:
        g = _get_github_client(request.user)
        repos = g.get_user().get_repos()
        return Response([{'full_name': r.full_name, 'private': r.private} for r in repos])
    except PermissionError as e:
        return Response({'error': str(e)}, status=403)
    except Exception as e:
        return Response({'error': str(e)}, status=500)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def github_push(request, project_id):
    """Push project files to a GitHub repository using the GitHub API."""
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response({"error": "Only the owner can push"}, status=403)

    repo_full_name = request.data.get('repo')  # e.g., "username/repo"
    branch = request.data.get('branch', 'main')
    commit_message = request.data.get('message', f'Update from NeuralOps at {timezone.now().isoformat()}')

    if not repo_full_name:
        return Response({"error": "repo (full_name) required"}, status=400)

    try:
        g = _get_github_client(request.user)
        repo = g.get_repo(repo_full_name)
    except GithubException as e:
        return Response({'error': f'Repository not accessible: {e}'}, status=404)

    # Get the current HEAD commit SHA of the branch
    try:
        ref = repo.get_git_ref(f'heads/{branch}')
        sha_latest_commit = ref.object.sha
        base_tree = repo.get_git_commit(sha_latest_commit).tree
    except GithubException:
        # If branch doesn't exist, create it from default branch or empty
        try:
            default_branch = repo.default_branch
            ref = repo.get_git_ref(f'heads/{default_branch}')
            sha_latest_commit = ref.object.sha
            base_tree = repo.get_git_commit(sha_latest_commit).tree
            # Create new branch
            repo.create_git_ref(f'refs/heads/{branch}', sha_latest_commit)
        except:
            # Empty repo? We'll create initial commit
            base_tree = None

    # Build tree from project files
    tree_list = []
    for f in project.files.all():
        content = f.content.encode('utf-8') if not f.is_binary else ''
        blob = repo.create_git_blob(content, 'utf-8')
        tree_list.append({
            'path': f.filename,
            'mode': '100644',
            'type': 'blob',
            'sha': blob.sha,
        })

    if tree_list:
        new_tree = repo.create_git_tree(tree_list, base_tree=base_tree)
        parent_commits = [repo.get_git_commit(sha_latest_commit)] if sha_latest_commit else []
        new_commit = repo.create_git_commit(
            message=commit_message,
            tree=new_tree,
            parents=parent_commits,
        )
        # Update the ref to point to the new commit
        try:
            ref.edit(sha=new_commit.sha)
        except:
            repo.create_git_ref(f'refs/heads/{branch}', new_commit.sha)
        return Response({"status": "pushed", "commit": new_commit.sha})
    else:
        return Response({"status": "no files to push"})

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def github_pull(request, project_id):
    """Pull files from a GitHub repository and update the project."""
    project = _get_accessible_project(project_id, request.user)
    if project.owner != request.user:
        return Response({"error": "Only the owner can pull"}, status=403)

    repo_full_name = request.data.get('repo')
    branch = request.data.get('branch', 'main')
    if not repo_full_name:
        return Response({"error": "repo required"}, status=400)

    try:
        g = _get_github_client(request.user)
        repo = g.get_repo(repo_full_name)
        contents = repo.get_contents('', ref=branch)
        updated = 0
        for content_file in contents:
            if content_file.type == 'file':
                try:
                    file_content = content_file.decoded_content.decode('utf-8')
                except:
                    file_content = ''
                PaperFile.objects.update_or_create(
                    project=project,
                    filename=content_file.path,
                    defaults={'content': file_content},
                )
                updated += 1
        project.version += 1
        project.save(update_fields=['version', 'updated_at'])
        return Response({"status": "pulled", "files_updated": updated})
    except GithubException as e:
        return Response({'error': str(e)}, status=500)