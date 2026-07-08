from django.urls import path

from . import views

urlpatterns = [
    # ── Questions ───────────────────────────────────────────
    path("questions/", views.question_list, name="question_list"),
    path(
        "questions/<uuid:question_id>/", views.question_detail, name="question_detail"
    ),
    path(
        "questions/<uuid:question_id>/close/",
        views.close_question,
        name="close_question",
    ),
    path(
        "questions/<uuid:question_id>/related/",
        views.related_questions,
        name="related_questions",
    ),
    # ── Answers ────────────────────────────────────────────
    path("questions/<uuid:question_id>/answer/", views.post_answer, name="post_answer"),
    path("answers/<uuid:answer_id>/", views.answer_detail, name="answer_detail"),
    path("answers/<uuid:answer_id>/accept/", views.accept_answer, name="accept_answer"),
    # ── Voting ─────────────────────────────────────────────
    path(
        "questions/<uuid:question_id>/vote/", views.vote_question, name="vote_question"
    ),
    path("answers/<uuid:answer_id>/vote/", views.vote_answer, name="vote_answer"),
    # ── Comments ───────────────────────────────────────────
    path(
        "questions/<uuid:question_id>/comment/",
        views.post_comment,
        name="comment_question",
    ),
    path(
        "answers/<uuid:answer_id>/comment/", views.post_comment, name="comment_answer"
    ),
    # ── Bounty ─────────────────────────────────────────────
    path(
        "questions/<uuid:question_id>/bounty/", views.start_bounty, name="start_bounty"
    ),
    # ── Bookmarks ──────────────────────────────────────────
    path(
        "questions/<uuid:question_id>/bookmark/",
        views.bookmark_question,
        name="bookmark",
    ),
    path(
        "questions/<uuid:question_id>/unbookmark/",
        views.unbookmark_question,
        name="unbookmark",
    ),
    # ── Tags ──────────────────────────────────────────────
    path("tags/", views.tag_list, name="tag_list"),
    path("tags/<slug:slug>/", views.tag_detail, name="tag_detail"),
    # ── User Profile ──────────────────────────────────────
    path(
        "user/<str:username>/reputation/", views.user_reputation, name="user_reputation"
    ),
    path("user/<str:username>/activity/", views.activity_graph, name="activity_graph"),
    # ── Leaderboard ────────────────────────────────────────
    path("leaderboard/", views.leaderboard, name="leaderboard"),
    # ── Badges ─────────────────────────────────────────────
    path("badges/", views.badge_list, name="badge_list"),
    path("badges/mine/", views.my_badges, name="my_badges"),
    # ── Privileges ─────────────────────────────────────────
    path("privileges/", views.privilege_list, name="privilege_list"),
    # ── Trust Level ───────────────────────────────────────
    path("trust-level/", views.my_trust_level, name="my_trust_level"),
    # ── Digest ────────────────────────────────────────────
    path("digest/", views.digest_settings, name="digest_settings"),
]
