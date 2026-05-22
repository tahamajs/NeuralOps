import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent.parent
CODES_DIR = BASE_DIR.parent

def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE or export KEY=VALUE lines without overriding env."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


# ----------------------------------------------------------------------
# Environment and import paths
# ----------------------------------------------------------------------
load_env_file(PROJECT_ROOT / ".env")
load_env_file(BASE_DIR / ".env")

# Make the Codes package root importable so `framework.*` resolves without
# shadowing Django apps such as `model_versioning`.
if str(CODES_DIR) not in sys.path:
    sys.path.insert(0, str(CODES_DIR))

from framework.utils.paths import get_glue_dir

# ----------------------------------------------------------------------
# Celery & Redis
# ----------------------------------------------------------------------
CELERY_BROKER_URL = 'redis://localhost:6379/0'
CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'

# ----------------------------------------------------------------------
# Security & Debug
# ----------------------------------------------------------------------
SECRET_KEY = os.environ.get("DJANGO_SECRET_KEY", "dev-secret-change-me")
DEBUG = os.environ.get("DJANGO_DEBUG", "1") == "1"
ALLOWED_HOSTS = ["*"] if DEBUG else os.environ.get("ALLOWED_HOSTS", "").split(",")
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')
OPENAI_BASE_URL = os.environ.get('OPENAI_BASE_URL', None)  # Optional, for compatible APIs

# ----------------------------------------------------------------------
# Application definition
# ----------------------------------------------------------------------
INSTALLED_APPS = [
    'daphne',
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
'computational_discovery',
'experiment_accelerator',

    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'channels',
    'django_filters',
'idea_tournament',

    'api',
    'web_dashboard',
    'experiment',
    'jobscheduler',
    'notes',
    'model_versioning',
    'terminal',
    'agents',
    'portfolios',
'gamification',

    'gpu_marketplace',
    'gpu_escrow',
    'gpu_validator',
    'gpu_benchmark',
    'gpu_federation',
    'gpu_finance',
    'gpu_reputation',
'streaks',

    'organizations',
    'community',
    'leaderboards',
    'reviews',
    'sweeps',
    'django.contrib.sites',        # required by allauth
    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.github',   # optional
'social_network',
'collaborator_matching',
'expert_wall',
'impact_predictor',
'literature_review',
'literature_discovery',
'ai_coscientist',
'anomaly_hpo',
'autonomous_research',
'marketplace_ai',
'coding_assistant',
'experiment_recommender',

    'profiles',
    'messaging',
'reputation',

    'paper',
]

AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
]

SITE_ID = 1
SOCIALACCOUNT_PROVIDERS = {
    'github': {
        'SCOPE': ['user:email', 'repo'],
        'VERIFIED_EMAIL': True,
    },
    'huggingface': {
        'SCOPE': ['openid', 'profile', 'email'],
    },
}


CELERY_BEAT_SCHEDULE.update({
    'regenerate-all-interest-profiles': {
        'task': 'collaborator_matching.tasks.regenerate_all_profiles',
        'schedule': crontab(hour=3, minute=0, day_of_week=1),  # Every Monday
    },
    'generate-all-matches': {
        'task': 'collaborator_matching.tasks.generate_all_matches',
        'schedule': crontab(hour=4, minute=0, day_of_week=1),
    },
})

CELERY_BEAT_SCHEDULE.update({
    'reset-weekly-points': {
        'task': 'gamification.tasks.reset_weekly_points',
        'schedule': crontab(hour=0, minute=0, day_of_week=1),  # Every Monday
    },
    'check-streak-decay': {
        'task': 'gamification.tasks.check_streak_decay',
        'schedule': crontab(hour=0, minute=10),
    },
    'take-leaderboard-snapshot': {
        'task': 'gamification.tasks.take_leaderboard_snapshot',
        'schedule': crontab(hour=0, minute=20),
    },
    'award-streak-bonus': {
        'task': 'gamification.tasks.award_streak_bonus',
        'schedule': crontab(hour=0, minute=30, day_of_week=1),  # Every Monday
    },
})

CELERY_BEAT_SCHEDULE.update({
    'generate-weekly-summaries': {
        'task': 'streaks.tasks.generate_weekly_summaries',
        'schedule': crontab(hour=8, minute=0, day_of_week=1),  # Every Monday 8 AM
    },
    'check-broken-streaks': {
        'task': 'streaks.tasks.check_and_reset_broken_streaks',
        'schedule': crontab(hour=2, minute=0),  # Daily at 2 AM
    },
})
SOCIALACCOUNT_STORE_TOKENS = True
CELERY_BEAT_SCHEDULE.update({
    'weekly-experiment-summary': {
        'task': 'experiment_accelerator.tasks.generate_weekly_summary',
        'schedule': crontab(hour=9, minute=0, day_of_week=1),  # Monday 9 AM
    },
})
# ----------------------------------------------
# 
# 
# ------------------------
# Celery Beat schedule (all combined)
# ----------------------------------------------------------------------
from celery.schedules import crontab

CELERY_BEAT_SCHEDULE = {
    # Marketplace
    'match-orders-every-30s': {
        'task': 'gpu_marketplace.tasks.run_matching_cycle',
        'schedule': 30.0,
    },
    'update-dynamic-pricing-every-5min': {
        'task': 'gpu_marketplace.tasks.update_dynamic_pricing',
        'schedule': crontab(minute='*/5'),
    },
    'expire-stale-orders-hourly': {
        'task': 'gpu_marketplace.tasks.expire_stale_orders',
        'schedule': crontab(minute=0, hour='*'),
    },
    'update-reliability-scores-daily': {
        'task': 'gpu_marketplace.tasks.update_reliability_scores',
        'schedule': crontab(minute=0, hour=3),
    },

    # Finance
    'daily-index-calculation': {
        'task': 'gpu_finance.tasks.daily_index_calculation',
        'schedule': crontab(hour=0, minute=5),
    },
    'settle-expired-futures': {
        'task': 'gpu_finance.tasks.settle_expired_futures',
        'schedule': crontab(hour=0, minute=15),
    },
    'generate-monthly-invoices': {
        'task': 'gpu_finance.tasks.generate_monthly_invoices',
        'schedule': crontab(day_of_month=1, hour=1, minute=0),
    },

    # Reputation
    'compute-daily-trust-graph': {
        'task': 'gpu_reputation.tasks.compute_daily_trust_graph',
        'schedule': crontab(hour=2, minute=0),
    },
    'auto-award-badges': {
        'task': 'gpu_reputation.tasks.auto_award_badges',
        'schedule': crontab(hour=1, minute=0),
    },
    'expire-stale-disputes': {
        'task': 'gpu_reputation.tasks.expire_stale_disputes',
        'schedule': crontab(hour=0, minute=30),
    },
}
CELERY_BEAT_SCHEDULE.update({
    'daily-reputation-snapshot': {
        'task': 'reputation.tasks.daily_snapshot',
        'schedule': crontab(hour=1, minute=0),
    },
    'generate-daily-recommendations': {
        'task': 'reputation.tasks.generate_daily_recommendations',
        'schedule': crontab(hour=2, minute=0),
    },
})
# Social auth backends — uncomment after installing django-allauth
# AUTHENTICATION_BACKENDS = [
#     'django.contrib.auth.backends.ModelBackend',
#     'allauth.account.auth_backends.AuthenticationBackend',
# ]

# SITE_ID = 1
# SOCIALACCOUNT_PROVIDERS = {
#     'github': {
#         'SCOPE': ['user:email', 'repo'],
#         'VERIFIED_EMAIL': True,
#     },
#     'huggingface': {
#         'SCOPE': ['openid', 'profile', 'email'],
#     },
# }
# SOCIALACCOUNT_STORE_TOKENS = True

# ----------------------------------------------------------------------
# Middleware
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'allauth.account.middleware.AccountMiddleware',         # ← add this
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'web_dashboard.middleware.TokenAuthMiddleware',
]

# ----------------------------------------------------------------------
# URL & WSGI / ASGI
# ----------------------------------------------------------------------
ROOT_URLCONF = "django_api.urls"
WSGI_APPLICATION = "django_api.wsgi.application"
ASGI_APPLICATION = "django_api.asgi.application"

# ----------------------------------------------------------------------
# Channels
# ----------------------------------------------------------------------
CHANNEL_LAYERS = {
    'default': {
        'BACKEND': 'channels.layers.InMemoryChannelLayer',
    },
}

# ----------------------------------------------------------------------
# Templates
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# Database
# ----------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ----------------------------------------------------------------------
# REST Framework
# ----------------------------------------------------------------------
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'api.authentication.CookieTokenAuthentication',
        'rest_framework.authentication.TokenAuthentication',
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
# CORS
# ----------------------------------------------------------------------
CORS_ALLOW_CREDENTIALS = True
if DEBUG:
    CORS_ALLOW_ALL_ORIGINS = True
else:
    CORS_ALLOW_ALL_ORIGINS = False
    CORS_ALLOWED_ORIGINS = os.environ.get("CORS_ALLOWED_ORIGINS", "").split(",")

# ----------------------------------------------------------------------
# Static & Media
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
# Custom Settings
# ----------------------------------------------------------------------
GLUE_DATA_PATH = os.environ.get(
    "GLUE_DATA_PATH",
    str(get_glue_dir()),
)

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