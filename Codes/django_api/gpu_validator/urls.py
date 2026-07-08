from django.urls import path

from . import views

urlpatterns = [
    # ── Validator Nodes ────────────────────────────────────────
    path("validators/register/", views.register_validator, name="register_validator"),
    path("validators/me/", views.my_validator_status, name="my_validator_status"),
    path("validators/me/update/", views.update_validator, name="update_validator"),
    path("validators/", views.validator_list, name="validator_list"),
    path("validators/<uuid:node_id>/", views.validator_detail, name="validator_detail"),
    # ── Verification Tasks ────────────────────────────────────
    path("tasks/", views.verification_task_list, name="verification_task_list"),
    path(
        "tasks/<uuid:task_id>/",
        views.verification_task_detail,
        name="verification_task_detail",
    ),
    path(
        "tasks/<uuid:task_id>/submit/",
        views.submit_verification,
        name="submit_verification",
    ),
    path(
        "tasks/<uuid:task_id>/finalize/",
        views.finalize_consensus,
        name="finalize_consensus",
    ),
    path("tasks/mine/", views.my_verification_tasks, name="my_verification_tasks"),
    # ── Proof of Compute ─────────────────────────────────────
    path(
        "proofs/<uuid:job_id>/",
        views.proof_of_compute_detail,
        name="proof_of_compute_detail",
    ),
    path("proofs/<uuid:job_id>/generate/", views.generate_proof, name="generate_proof"),
]
