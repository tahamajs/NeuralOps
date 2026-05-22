from django.urls import path
from . import views

urlpatterns = [
    # Dashboards
    path('dashboards/', views.dashboard_list, name='dashboard_list'),
    path('dashboards/<uuid:dashboard_id>/', views.dashboard_detail, name='dashboard_detail'),
    path('dashboards/<uuid:dashboard_id>/public/', views.public_dashboard, name='public_dashboard'),

    # Widgets
    path('dashboards/<uuid:dashboard_id>/widgets/', views.add_widget, name='add_widget'),
    path('widgets/<uuid:widget_id>/', views.widget_detail, name='widget_detail'),
    path('dashboards/<uuid:dashboard_id>/layout/', views.save_layout, name='save_layout'),

    # SQL
    path('sql/queries/', views.sql_query_list, name='sql_query_list'),
    path('sql/execute/', views.execute_sql, name='execute_sql'),

    # Live data
    path('live-experiments/', views.live_experiment_data, name='live_experiment_data'),
]