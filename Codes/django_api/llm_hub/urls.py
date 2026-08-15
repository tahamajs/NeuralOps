from django.urls import path

from . import views

urlpatterns = [
    path("", views.provider_list, name="llm_hub_root"),
    path("providers/", views.provider_list, name="llm_providers"),
    path("models/", views.model_list, name="llm_models"),
    path("playground/chat/", views.playground_chat, name="playground_chat"),
    path("playground/compare/", views.playground_compare, name="playground_compare"),
    path("prompts/", views.prompt_list, name="prompt_list"),
    path("prompts/<uuid:prompt_id>/", views.prompt_detail, name="prompt_detail"),
    path(
        "prompts/<uuid:prompt_id>/version/",
        views.prompt_new_version,
        name="prompt_new_version",
    ),
    path("eval/suites/", views.eval_suite_list, name="eval_suite_list"),
    path("eval/start/", views.start_evaluation, name="start_evaluation"),
    path("eval/runs/<uuid:run_id>/", views.eval_run_detail, name="eval_run_detail"),
    path("cost/", views.cost_dashboard, name="cost_dashboard"),
    path("memory/", views.memory_list, name="memory_list"),
    path("memory/context/", views.memory_context, name="memory_context"),
    path("tools/", views.tool_list, name="tool_list"),
    path("schemas/", views.schema_list, name="schema_list"),
    path("arena/battle/", views.get_battle, name="arena_battle"),
    path("arena/battle/<uuid:battle_id>/vote/", views.submit_vote, name="arena_vote"),
    path("arena/leaderboard/", views.arena_leaderboard, name="arena_leaderboard"),
    path("guardrails/", views.create_guardrail, name="create_guardrail"),
    path(
        "guardrails/attach/<uuid:session_id>/",
        views.attach_guardrail,
        name="attach_guardrail",
    ),
    path(
        "guardrails/test/<uuid:guardrail_id>/",
        views.test_guardrail,
        name="test_guardrail",
    ),
]
