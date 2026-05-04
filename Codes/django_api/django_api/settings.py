import os, sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Make the outer framework importable
sys.path.insert(0, str(BASE_DIR.parent))


# HUGGINGFACE_TOKEN = os.environ.get('HF_TOKEN')
# HUGGINGFACE_USERNAME = os.environ.get('HF_USERNAME')  # Your HF username or org
# MODEL_PUBLISH_BUCKET = 'your-bucket-name'  # For intermediate storage



# SCHEDULER_TOTAL_GPUS = int(os.environ.get('SCHEDULER_TOTAL_GPUS', 1))
# SCHEDULER_BOOST_WAIT_THRESHOLD = int(os.environ.get('SCHEDULER_BOOST_WAIT_THRESHOLD', 600))
# SCHEDULER_SHORT_JOB_THRESHOLD = int(os.environ.get('SCHEDULER_SHORT_JOB_THRESHOLD', 60))


SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"]
INSTALLED_APPS = [
    'daphne',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.staticfiles',
    'rest_framework',
    'corsheaders',
    'channels',
    'api',          # ← appears only once
    'rest_framework.authtoken',
    'web_dashboard',
]
ASGI_APPLICATION = 'django_api.asgi.application'
# LOGIN_URL = '/login/'

MIDDLEWARE = [
    #     'web_dashboard.middleware.WellKnownIgnoreMiddleware',   # <-- add this line

    # 'web_dashboard.middleware.TokenAuthMiddleware',  # <-- add this line at the top
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]
ROOT_URLCONF = "django_api.urls"
WSGI_APPLICATION = "django_api.wsgi.application"
# ASGI_APPLICATION = "django_api.asgi.application"

# Channels layer (in‑memory for dev; use Redis in production)
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
        # 'api.authentication.CookieTokenAuthentication',   # <-- use custom class
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
    ],
}
LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

CORS_ALLOW_ALL_ORIGINS = True
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],  # optional, for global templates
        'APP_DIRS': True,  # this tells Django to look inside each app's templates/ folder
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
LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
    
}