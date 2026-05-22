from django.urls import path
from . import views

urlpatterns = [
    # Agent Mode
    path('query/', views.agent_query, name='agent_query'),

    # Comparisons
    path('compare/', views.comparison_list, name='comparison_list'),
    path('compare/<uuid:comparison_id>/', views.comparison_detail, name='comparison_detail'),

    # Compliance
    path('compliance/<uuid:listing_id>/', views.check_compliance_view, name='check_compliance'),
    path('compliance/history/', views.compliance_history, name='compliance_history'),

    # Recommendations
    path('recommendations/', views.smart_recommendations, name='smart_recommendations'),
    path('recommendations/generate/', views.generate_recommendations, name='generate_recommendations'),
    path('recommendations/<uuid:recommendation_id>/dismiss/', views.dismiss_recommendation, name='dismiss_recommendation'),
]