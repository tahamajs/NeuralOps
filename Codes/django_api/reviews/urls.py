# reviews/urls.py
from django.urls import path
from . import views

urlpatterns = [
    # ── Review Requests ──────────────────────────────────────
    path('requests/', views.review_request_list, name='review_request_list'),
    path('requests/<uuid:request_id>/', views.review_request_detail, name='review_request_detail'),
    path('requests/<uuid:request_id>/submit/', views.submit_for_review, name='submit_for_review'),
    path('requests/<uuid:request_id>/open/', views.open_for_review, name='open_for_review'),
    path('requests/<uuid:request_id>/complete/', views.complete_review_round, name='complete_review_round'),
    path('requests/<uuid:request_id>/summary/', views.review_summary, name='review_summary'),

    # ── Reviews ──────────────────────────────────────────────
    path('requests/<uuid:request_id>/review/', views.write_review, name='write_review'),
    path('my-reviews/', views.my_reviews_given, name='my_reviews_given'),

    # ── Comments ─────────────────────────────────────────────
    path('reviews/<uuid:review_id>/comment/', views.add_review_comment, name='add_review_comment'),
    path('comments/<uuid:comment_id>/resolve/', views.resolve_comment, name='resolve_comment'),

    # ── Helpfulness Voting ──────────────────────────────────
    path('reviews/<uuid:review_id>/vote/', views.vote_review_helpful, name='vote_review_helpful'),

    # ── Templates ────────────────────────────────────────────
    path('templates/', views.template_list, name='review_template_list'),

    # ── Badges ───────────────────────────────────────────────
    path('badges/', views.badge_list, name='review_badge_list'),
    path('badges/mine/', views.my_badges, name='my_reviewer_badges'),

    # ── Stats ────────────────────────────────────────────────
    path('stats/', views.global_stats, name='review_stats'),
]