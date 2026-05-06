# django_api/settings.py
import os
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# Base directory
# ----------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

# Make the 'Codes' folder (where all Django apps reside) importable
CODES_DIR = BASE_DIR.parent / 'Codes'
if CODES_DIR.exists() and str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

# Also add the parent of Codes (Project root) if needed by the framework
PROJECT_ROOT = BASE_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    'daphne',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',

    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'channels',
    'django_filters',

    # Your project apps (now accessible because Codes/ is on sys.path)
    'api',
    'web_dashboard',
    'experiment',   # not 'experiments'

    'jobscheduler',
    'notes',                 # this will work if Codes/notes exists
    'model_versioning',
    
]

# ----------------------------------------------------------------------
# (Optional) Remove 'notes' if the directory does not exist
# ----------------------------------------------------------------------
# if not (CODES_DIR / 'notes').exists():
#     INSTALLED_APPS.remove('notes')

# ----------------------------------------------------------------------
# Middleware
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
    'web_dashboard.middleware.TokenAuthMiddleware',

]

# ----------------------------------------------------------------------
# URL configuration
# ----------------------------------------------------------------------
ROOT_URLCONF = "django_api.urls"

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
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

WSGI_APPLICATION = "django_api.wsgi.application"
ASGI_APPLICATION = "django_api.asgi.application"

CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer',
    },
}

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'api.authentication.CookieTokenAuthentication',
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

CORS_ALLOW_ALL_ORIGINS = DEBUG
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [] if DEBUG else os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",")

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

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

HUGGINGFACE_TOKEN = os.environ.get("HF_TOKEN")
HUGGINGFACE_USERNAME = os.environ.get("HF_USERNAME")
MODEL_ARCHIVE_BUCKET = os.environ.get("MODEL_ARCHIVE_BUCKET")

SCHEDULER_TOTAL_GPUS = int(os.environ.get("SCHEDULER_TOTAL_GPUS", 1))
SCHEDULER_BOOST_WAIT_THRESHOLD = int(os.environ.get("SCHEDULER_BOOST_WAIT_THRESHOLD", 600))
SCHEDULER_SHORT_JOB_THRESHOLD = int(os.environ.get("SCHEDULER_SHORT_JOB_THRESHOLD", 60))
SCHEDULER_PREEMPT_TIMEOUT = int(os.environ.get("SCHEDULER_PREEMPT_TIMEOUT", 10))
JOB_TIMEOUT_SECONDS = int(os.environ.get("JOB_TIMEOUT_SECONDS", 0))

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
ML_TOKEN = os.environ.get("ML_TOKEN", "")
LOGIN_URL = "/login/"
ADMIN_SITE_HEADER = "Transformer Toolkit Admin"