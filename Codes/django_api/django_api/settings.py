# django_api/settings.py
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Base directory
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Make the outer framework (e.g., parent folder containing `framework`) importable
sys.path.insert(0, str(BASE_DIR.parent))

# ----------------------------------------------------------------------
# Security & Debug
# ----------------------------------------------------------------------
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"] if DEBUG else os.environ.get("ALLOWED_HOSTS", "").split(",")

# ----------------------------------------------------------------------
# Installed apps
# ----------------------------------------------------------------------
INSTALLED_APPS = [
    # First-party Django apps
    'daphne',                           # ASGI server (must be first)
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    # Third-party
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'channels',
    'django_filters',

    # Project apps
    'api',                              # authentication, system status, config endpoints
    'web_dashboard',                    # HTML pages (dashboard, login, etc.)
    'experiments',                      # Experiment & TrainingRun models, API
    'jobscheduler',                     # Job scheduler, GPU management, custom metrics
    'notes',                            # Researcher notebook (with versioning, webhooks)
    'model_versioning',                 # Model versioning and Hugging Face Hub integration
]

# ----------------------------------------------------------------------
# Middleware (order matters)
# ----------------------------------------------------------------------
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    # Optional: custom middleware for token auth on HTML pages and .well‑known
    # 'web_dashboard.middleware.TokenAuthMiddleware',          # uncomment to protect pages
    # 'web_dashboard.middleware.WellKnownIgnoreMiddleware',    # uncomment to suppress devtools probe
]

# ----------------------------------------------------------------------
# URL configuration
# ----------------------------------------------------------------------
ROOT_URLCONF = "django_api.urls"

# ----------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------
TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],      # global templates directory
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

# ----------------------------------------------------------------------
# WSGI / ASGI
# ----------------------------------------------------------------------
WSGI_APPLICATION = "django_api.wsgi.application"
ASGI_APPLICATION = "django_api.asgi.application"

# Channels layer (in‑memory for development; use Redis in production)
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer',
    },
}

# ----------------------------------------------------------------------
# Database (SQLite for development – can switch to PostgreSQL)
# ----------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ----------------------------------------------------------------------
# Authentication & REST Framework
# ----------------------------------------------------------------------
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'api.authentication.CookieTokenAuthentication',   # token from header or cookie
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
    ],
    'DEFAULT_FILTER_BACKENDS': [
        'django_filters.rest_framework.DjangoFilterBackend',
    ],
    'DEFAULT_PAGINATION_CLASS': 'rest_framework.pagination.PageNumberPagination',
    'PAGE_SIZE': 20,
}

# ----------------------------------------------------------------------
# CORS (allow all for development; restrict in production)
# ----------------------------------------------------------------------
CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [] if DEBUG else os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",")

# ----------------------------------------------------------------------
# Static & Media files
# ----------------------------------------------------------------------
STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

# ----------------------------------------------------------------------
# Internationalization
# ----------------------------------------------------------------------
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {message}",
            "style": "{",
        },
        "json": {
            "format": '{"time": "%(asctime)s", "level": "%(levelname)s", "name": "%(name)s", "message": "%(message)s"}',
        },
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "verbose" if DEBUG else "json",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

# ----------------------------------------------------------------------
# Optional: Hugging Face Hub and S3 Cold Storage
# ----------------------------------------------------------------------
HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_USERNAME = os.environ.get("HF_USERNAME")
MODEL_ARCHIVE_BUCKET = os.environ.get("MODEL_ARCHIVE_BUCKET")   # for S3 Glacier

# ----------------------------------------------------------------------
# Scheduler settings (optional overrides)
# ----------------------------------------------------------------------
SCHEDULER_TOTAL_GPUS = int(os.environ.get("SCHEDULER_TOTAL_GPUS", 1))
SCHEDULER_BOOST_WAIT_THRESHOLD = int(os.environ.get("SCHEDULER_BOOST_WAIT_THRESHOLD", 600))
SCHEDULER_SHORT_JOB_THRESHOLD = int(os.environ.get("SCHEDULER_SHORT_JOB_THRESHOLD", 60))
SCHEDULER_PREEMPT_TIMEOUT = int(os.environ.get("SCHEDULER_PREEMPT_TIMEOUT", 10))
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_SECONDS", 0))

# ----------------------------------------------------------------------
# API base URL (used by logger.py)
# ----------------------------------------------------------------------
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

# ----------------------------------------------------------------------
# ML Token for logger (if needed)
# ----------------------------------------------------------------------
ML_TOKEN = os.environ.get("ML_TOKEN", "")

# ----------------------------------------------------------------------
# Login URL (for decorator @login_required, not used with token auth)
# ----------------------------------------------------------------------
LOGIN_URL = "/login/"

# ----------------------------------------------------------------------
# Django Admin
# ----------------------------------------------------------------------
ADMIN_SITE_HEADER = "Transformer Toolkit Admin"