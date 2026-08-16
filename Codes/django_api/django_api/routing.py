"""
WebSocket routing aggregator for the Django project.
Collects and exposes websocket_urlpatterns from all installed apps.
"""

import logging

logger = logging.getLogger(__name__)


def _safe_import_ws(module_path):
    try:
        mod = __import__(module_path, fromlist=["websocket_urlpatterns"])
        return getattr(mod, "websocket_urlpatterns", [])
    except (ImportError, AttributeError) as exc:
        logger.debug("Could not load websocket routes from %s: %s", module_path, exc)
        return []


# App routing modules in prioritized order
ROUTING_MODULES = [
    "terminal.routing",
    "research_agent.routing",
    "agents.routing",
    "messaging.routing",
    "paper.routing",
    "notes.routing",
    "coding_assistant.routing",
    "research_coach.routing",
    "llm_hub.routing",
    "gpu_marketplace.routing",
    "health_monitor.routing",
    "support.routing",
    "web_dashboard.routing",
    "experiment.routing",
    "jobscheduler.routing",
    "api.routing",
    "community.routing",
]

websocket_urlpatterns = []
for _mod_name in ROUTING_MODULES:
    websocket_urlpatterns.extend(_safe_import_ws(_mod_name))
