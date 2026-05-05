# Backend Integration Guide

## Quick Start

### 1. Install Dependencies
```bash
cd Codes/django_api
pip install -r requirements.txt
```

### 2. Database Setup
```bash
python manage.py migrate
python manage.py createsuperuser  # Optional: for admin panel
```

### 3. Run Development Server
```bash
# Using Daphne (ASGI server with WebSocket support)
daphne -b 0.0.0.0 -p 8000 django_api.asgi:application

# OR using Django development server (no WebSocket support)
python manage.py runserver 0.0.0.0:8000
```

## API Endpoints

### Authentication
- `POST /api/auth/register/` - Register new user
- `POST /api/auth/login/` - Login and get token
- `POST /api/auth/logout/` - Logout
- `GET /api/auth/user/` - Get current user info
- `PATCH /api/auth/profile/` - Update profile
- `POST /api/auth/change-password/` - Change password
- `POST /api/auth/refresh-token/` - Refresh token

### Configuration
- `GET /api/config/models/` - List available models
- `GET /api/config/tasks/` - List available tasks
- `GET /api/config/methods/` - List compression methods
- `GET /api/config/attentions/` - List attention types
- `GET /api/config/presets/` - List training presets

### System
- `GET /api/system/status/` - System resource usage (CPU, RAM, disk)
- `GET /api/system/gpu/` - GPU information
- `GET /api/health/` - Health check endpoint
- `GET /api/version/` - API version info

### Experiments
- `GET /api/experiments/` - List all experiments
- `POST /api/experiments/` - Create experiment
- `GET /api/experiments/{id}/` - Get experiment details
- `PATCH /api/experiments/{id}/` - Update experiment
- `DELETE /api/experiments/{id}/` - Delete experiment

### Jobs
- `POST /api/train/` - Start training job
- `POST /api/evaluate/` - Start evaluation job
- `POST /api/compress/` - Start compression job
- `GET /api/runs/{id}/` - Get job details

### WebSocket
- `WS /ws/experiments/{experiment_id}/` - Live experiment updates
- `WS /ws/jobs/{job_id}/` - Live job updates
- `WS /ws/terminal/` - Terminal output stream

## Environment Variables

```bash
# Security
DJANGO_SECRET_KEY=your-secret-key
DJANGO_DEBUG=1  # Set to 0 in production

# Allowed hosts (comma-separated)
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# HuggingFace (optional)
HF_TOKEN=your-hf-token
HF_USERNAME=your-username

# GPU Settings
SCHEDULER_TOTAL_GPUS=1

# CORS (for frontend requests)
CORS_ALLOW_ALL_ORIGINS=1  # Set to 0 in production and specify CORS_ALLOWED_ORIGINS
```

## Authentication Flow

### Token-Based Authentication
1. User calls `POST /api/auth/login/` with username and password
2. Backend returns token
3. Frontend stores token in localStorage
4. Frontend includes token in `Authorization: Bearer {token}` header for subsequent requests

### Request Example
```bash
curl -X GET http://localhost:8000/api/experiments/ \
  -H "Authorization: Bearer your-token-here"
```

## Common Issues

### "CORS Error" or "Cross-origin request blocked"
- Ensure `CORS_ALLOW_ALL_ORIGINS=1` in development
- Check that frontend URL matches `ALLOWED_HOSTS`

### "Unauthorized" (401)
- Token may have expired
- Try refreshing token: `POST /api/auth/refresh-token/`
- Check that token is correctly formatted in header

### WebSocket Connection Fails
- Ensure server is running with Daphne (not basic Django server)
- Check that Channels is installed: `pip install channels daphne`
- Verify WebSocket URL matches: `ws://localhost:8000/ws/experiments/{id}/`

### Database Locked Error
- Remove old lock files: `rm -f *.db-wal *.db-shm`
- Run migrations again: `python manage.py migrate`

## Development Workflow

### 1. Make Changes to Models
```bash
python manage.py makemigrations
python manage.py migrate
```

### 2. Create/Update Admin Views
Edit `api/admin.py` to register your models

### 3. Test Endpoints
```bash
python manage.py test api
```

### 4. Access Admin Panel
Navigate to `http://localhost:8000/admin/`
Login with superuser credentials

## Production Deployment

### Environment Setup
```bash
export DJANGO_DEBUG=0
export DJANGO_SECRET_KEY=$(python -c 'import secrets; print(secrets.token_urlsafe(50))')
export ALLOWED_HOSTS=your-domain.com,api.your-domain.com
export CORS_ALLOWED_ORIGINS=https://your-frontend-domain.com
```

### Run Server
```bash
gunicorn -w 4 -b 0.0.0.0:8000 django_api.wsgi:application
```

### Use Daphne for WebSocket Support
```bash
daphne -b 0.0.0.0 -p 8000 django_api.asgi:application
```

## Testing

### Run All Tests
```bash
python manage.py test
```

### Test Specific App
```bash
python manage.py test api
```

### Test with Coverage
```bash
pip install coverage
coverage run --source='.' manage.py test
coverage report
```

## Useful Commands

```bash
# Create superuser
python manage.py createsuperuser

# Shell access
python manage.py shell

# Clear database
python manage.py flush

# Collect static files
python manage.py collectstatic --noinput

# Check for issues
python manage.py check
```
