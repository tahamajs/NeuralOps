from django.urls import path
from . import views

urlpatterns = [
    # ── Chat sessions ──────────────────────────
    path('chats/', views.ChatSessionListCreateView.as_view(), name='chat_sessions'),
    path('chats/<uuid:session_id>/', views.ChatSessionDetailView.as_view(), name='chat_session_detail'),
    path('chats/<uuid:session_id>/messages/', views.MessageListView.as_view(), name='chat_messages'),
    path('chats/<uuid:session_id>/send/', views.send_message, name='send_message'),
    path('chats/<uuid:session_id>/summary/', views.api_conversation_summary, name='conversation_summary'),
    path('chats/<uuid:session_id>/export/', views.api_export_conversation, name='export_conversation'),

    # ── Paper research & library ───────────────
    path('papers/search/', views.api_search_papers, name='search_papers'),
    path('papers/save/', views.api_save_paper, name='save_paper'),
    path('papers/saved/', views.api_saved_papers_list, name='saved_papers'),
    path('papers/saved/<uuid:paper_id>/', views.api_paper_detail, name='paper_detail'),
    path('papers/saved/<uuid:paper_id>/annotate/', views.api_annotate_paper, name='annotate_paper'),
    path('papers/saved/<uuid:paper_id>/delete/', views.api_delete_paper, name='delete_paper'),
    path('papers/library/', views.api_search_library, name='search_library'),
    path('papers/citation-graph/', views.api_citation_graph, name='citation_graph'),
    path('papers/citation-graph/edge/', views.api_add_citation_edge, name='add_citation_edge'),
    path('papers/bulk-import/', views.api_bulk_import_papers, name='bulk_import_papers'),

    # ── Paper recommendations ──────────────────
    path('papers/recommendations/<uuid:run_id>/', views.paper_recommendations_for_run, name='paper_recommendations_for_run'),
    path('papers/recommendations/library/', views.paper_recommendations_from_library, name='paper_recommendations_library'),
    path('papers/search-for-run/<uuid:run_id>/', views.search_papers_for_run_topic, name='search_papers_for_run'),

    # ── Training analysis ──────────────────────
    path('training/analyze/', views.api_analyze_run, name='analyze_run'),
    path('training/analysis/<uuid:run_id>/', views.api_run_analysis, name='run_analysis'),
    path('training/suggest-next/', views.api_suggest_next, name='suggest_next'),

    # ── Training monitor ───────────────────────
    path('monitor/start/', views.start_training_monitor, name='start_monitor'),

    # ── Prompt testing ─────────────────────────
    path('prompts/test/', views.api_test_prompt, name='test_prompt'),
]