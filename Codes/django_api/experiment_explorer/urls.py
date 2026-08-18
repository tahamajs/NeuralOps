from django.urls import path
from . import views

urlpatterns = [
    path("", views.dataset_list, name="experiment_explorer_root"),
    # Hyperparameter Explorer Endpoints
    path("experiments/", views.explorer_experiments, name="explorer_experiments"),
    path("correlations/", views.explorer_correlations, name="explorer_correlations"),
    # Datasets
    path("datasets/", views.dataset_list, name="dataset_list"),
    path("datasets/<uuid:dataset_id>/", views.dataset_detail, name="dataset_detail"),
    path("datasets/upload-csv/", views.upload_csv, name="upload_csv"),
    path("datasets/import-runs/", views.import_from_training_runs, name="import_from_runs"),
    path("datasets/<uuid:dataset_id>/visualizations/", views.visualization_list, name="visualization_list"),
    path("visualizations/<uuid:viz_id>/", views.visualization_delete, name="visualization_delete"),
]
