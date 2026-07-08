from django.urls import path

from . import views

urlpatterns = [
    path("sweeps/", views.sweep_list, name="sweep_list"),
    path("sweeps/<uuid:sweep_id>/", views.sweep_detail, name="sweep_detail"),
    path("sweeps/<uuid:sweep_id>/trials/", views.sweep_trials, name="sweep_trials"),
    path(
        "sweeps/<uuid:sweep_id>/trials/<int:trial_number>/",
        views.trial_detail,
        name="trial_detail",
    ),
    path(
        "sweeps/<uuid:sweep_id>/trials/<int:trial_number>/update/",
        views.trial_update,
        name="trial_update",
    ),
    path(
        "sweeps/<uuid:sweep_id>/trials/<int:trial_number>/intermediate/",
        views.report_intermediate_value,
        name="report_intermediate",
    ),
    path(
        "sweeps/<uuid:sweep_id>/preference/",
        views.submit_preference,
        name="submit_preference",
    ),
    path(
        "sweeps/<uuid:sweep_id>/importance/",
        views.sweep_importance,
        name="sweep_importance",
    ),
    path("sweeps/<uuid:sweep_id>/pareto/", views.sweep_pareto, name="sweep_pareto"),
    path("sweeps/<uuid:sweep_id>/contour/", views.sweep_contour, name="sweep_contour"),
    path("sweeps/<uuid:sweep_id>/history/", views.sweep_history, name="sweep_history"),
    path("sweeps/<uuid:sweep_id>/slice/", views.sweep_slice, name="sweep_slice"),
    path(
        "sweeps/<uuid:sweep_id>/parallel/", views.sweep_parallel, name="sweep_parallel"
    ),
    path("sweeps/<uuid:sweep_id>/edf/", views.sweep_edf, name="sweep_edf"),
    path(
        "sweeps/<uuid:sweep_id>/warm-start/",
        views.sweep_warm_start,
        name="sweep_warm_start",
    ),
    path(
        "sweeps/<uuid:sweep_id>/dataframe/",
        views.sweep_dataframe,
        name="sweep_dataframe",
    ),
    path("sweeps/<uuid:sweep_id>/export/", views.sweep_export, name="sweep_export"),
    path("sweeps/import/", views.sweep_import, name="sweep_import"),
]
