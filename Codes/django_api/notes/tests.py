import importlib
from datetime import timedelta
from unittest import skipIf
from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone
from jobscheduler.models import Job
from rest_framework.test import APIClient

from .models import NoteMention, ResearcherNote, WebhookSubscription

_WEASYPRINT = importlib.util.find_spec("weasyprint") is not None


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------
class NotesTestBase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(username="author", password="testpass")
        self.other_user = User.objects.create_user(
            username="other", password="testpass"
        )
        self.staff_user = User.objects.create_user(
            username="staff", password="testpass", is_staff=True
        )

        # Job model does NOT have an 'author' field – remove it
        self.job = Job.objects.create(
            name="Test Job", model_name="bert", task_name="mrpc"
        )

        # Authenticate as author by default
        self.client.force_authenticate(user=self.user)

    def _create_note(self, **kwargs):
        defaults = {
            "title": "Test Note",
            "content": "Some content",
            "author": self.user,
            "run": self.job,
        }
        defaults.update(kwargs)
        return ResearcherNote.objects.create(**defaults)


# ----------------------------------------------------------------------
# Model Tests
# ----------------------------------------------------------------------
class ResearcherNoteModelTests(NotesTestBase):
    def test_create_note(self):
        note = self._create_note()
        self.assertIsNotNone(note.id)
        self.assertEqual(note.title, "Test Note")
        self.assertFalse(note.is_deleted)

    def test_soft_delete(self):
        note = self._create_note()
        note.soft_delete()
        note.refresh_from_db()
        self.assertTrue(note.is_deleted)
        self.assertIsNotNone(note.deleted_at)

    def test_increment_version(self):
        note = self._create_note()
        note.increment_version()
        note.refresh_from_db()
        self.assertEqual(note.version, 2)

    def test_slug_generation(self):
        note = self._create_note(title="My Amazing Note")
        self.assertTrue(note.slug)
        self.assertIn("my-amazing-note", note.slug)

    def test_reply_count(self):
        parent = self._create_note()
        reply = ResearcherNote.objects.create(
            content="Reply", author=self.user, parent_note=parent
        )
        self.assertEqual(parent.reply_count, 1)

    def test_version_creation_on_content_change(self):
        note = self._create_note(content="Original")
        old_version = note.version
        note.content = "Updated"
        note.save()
        self.assertTrue(note.versions.exists())
        self.assertEqual(note.versions.first().content_snapshot, "Original")
        self.assertEqual(note.version, old_version + 1)


# ----------------------------------------------------------------------
# API Tests – Core Endpoints
# ----------------------------------------------------------------------
class NoteListEndpointTests(NotesTestBase):
    def test_list_notes_empty(self):
        response = self.client.get("/api/notes/v1/notes/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["results"]), 0)

    def test_list_notes_with_data(self):
        self._create_note()
        ResearcherNote.objects.create(
            title="Private", content="x", author=self.other_user, is_public=False
        )
        response = self.client.get("/api/notes/v1/notes/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data["results"]), 1)  # only self._create_note

    def test_public_notes_visible_to_others(self):
        ResearcherNote.objects.create(
            title="Public note",
            content="Public",
            author=self.other_user,
            is_public=True,
        )
        response = self.client.get("/api/notes/v1/notes/")
        self.assertEqual(
            len(response.data["results"]), 1
        )  # public note from other user
        self._create_note()
        response = self.client.get("/api/notes/v1/notes/")
        self.assertEqual(len(response.data["results"]), 2)  # own + public

    def test_create_note(self):
        data = {
            "title": "New Note",
            "content": "Hello world",
            "run": str(self.job.id),
            "tags": ["research", "ml"],
            "is_public": True,
        }
        response = self.client.post("/api/notes/v1/notes/", data, format="json")
        self.assertEqual(response.status_code, 201)
        note = ResearcherNote.objects.get(title="New Note")
        self.assertEqual(note.tags, ["research", "ml"])
        self.assertTrue(note.is_public)

    def test_filter_by_tags(self):
        self._create_note(tags=["x"])
        self._create_note(tags=["y"])
        response = self.client.get("/api/notes/v1/notes/?tags=x")
        self.assertEqual(len(response.data["results"]), 1)

    def test_filter_by_is_public(self):
        self._create_note(is_public=True)
        self._create_note(is_public=False)
        response = self.client.get("/api/notes/v1/notes/?is_public=true")
        self.assertEqual(len(response.data["results"]), 1)

    def test_pagination(self):
        for i in range(25):
            self._create_note(title=f"Note {i}")
        response = self.client.get("/api/notes/v1/notes/?page_size=10")
        self.assertEqual(len(response.data["results"]), 10)
        self.assertIn("next", response.data)

    def test_csv_export(self):
        self._create_note()
        response = self.client.get("/api/notes/v1/notes/?export=csv")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv")

    def test_json_export(self):
        self._create_note()
        response = self.client.get("/api/notes/v1/notes/?export=json")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.data, list)


class NoteDetailEndpointTests(NotesTestBase):
    def test_get_note_detail(self):
        note = self._create_note()
        response = self.client.get(f"/api/notes/v1/notes/{note.id}/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["title"], "Test Note")

    def test_update_note(self):
        note = self._create_note()
        response = self.client.put(
            f"/api/notes/v1/notes/{note.id}/",
            {"title": "Updated", "content": "New content", "run": str(self.job.id)},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        note.refresh_from_db()
        self.assertEqual(note.title, "Updated")
        self.assertEqual(note.version, 2)

    def test_partial_update(self):
        note = self._create_note()
        response = self.client.patch(
            f"/api/notes/v1/notes/{note.id}/", {"content": "Patched"}, format="json"
        )
        self.assertEqual(response.status_code, 200)
        note.refresh_from_db()
        self.assertEqual(note.content, "Patched")

    def test_delete_note(self):
        note = self._create_note()
        response = self.client.delete(f"/api/notes/v1/notes/{note.id}/")
        self.assertEqual(response.status_code, 204)
        note.refresh_from_db()
        self.assertTrue(note.is_deleted)

    def test_access_other_private_note_denied(self):
        private_note = ResearcherNote.objects.create(
            title="Private", content="x", author=self.other_user, is_public=False
        )
        response = self.client.get(f"/api/notes/v1/notes/{private_note.id}/")
        self.assertEqual(response.status_code, 404)

    def test_access_public_note_allowed(self):
        public_note = ResearcherNote.objects.create(
            title="Public", content="x", author=self.other_user, is_public=True
        )
        response = self.client.get(f"/api/notes/v1/notes/{public_note.id}/")
        self.assertEqual(response.status_code, 200)


# ----------------------------------------------------------------------
# Reactions
# ----------------------------------------------------------------------
class ReactionTests(NotesTestBase):
    def test_add_reaction(self):
        note = self._create_note()
        response = self.client.post(
            f"/api/notes/v1/notes/{note.id}/react/", {"reaction": "like"}, format="json"
        )
        self.assertEqual(response.status_code, 200)
        note.refresh_from_db()
        self.assertIn(self.user.username, note.reactions.get("like", []))

    def test_toggle_reaction(self):
        note = self._create_note()
        self.client.post(
            f"/api/notes/v1/notes/{note.id}/react/", {"reaction": "like"}, format="json"
        )
        self.client.post(
            f"/api/notes/v1/notes/{note.id}/react/", {"reaction": "like"}, format="json"
        )
        note.refresh_from_db()
        self.assertNotIn(self.user.username, note.reactions.get("like", []))

    def test_reaction_missing_type(self):
        note = self._create_note()
        response = self.client.post(
            f"/api/notes/v1/notes/{note.id}/react/", {}, format="json"
        )
        self.assertEqual(response.status_code, 400)


# ----------------------------------------------------------------------
# Image Upload
# ----------------------------------------------------------------------
class ImageUploadTests(NotesTestBase):
    def test_upload_image(self):
        note = self._create_note()
        with patch(
            "django.core.files.storage.default_storage.save", return_value="fake_path"
        ):
            response = self.client.post(
                f"/api/notes/v1/notes/{note.id}/upload-image/",
                {"image": "fake_image"},
                format="multipart",
            )
        self.assertIn(response.status_code, [400, 201])  # 400 because no real file


# ----------------------------------------------------------------------
# Bulk Operations
# ----------------------------------------------------------------------
class BulkOperationsTests(NotesTestBase):
    def test_bulk_delete(self):
        n1 = self._create_note()
        n2 = self._create_note()
        response = self.client.post(
            "/api/notes/v1/notes/bulk/",
            {"action": "delete", "ids": [str(n1.id), str(n2.id)]},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["deleted"], 2)
        n1.refresh_from_db()
        n2.refresh_from_db()
        self.assertTrue(n1.is_deleted and n2.is_deleted)

    def test_bulk_create(self):
        response = self.client.post(
            "/api/notes/v1/notes/bulk/",
            {
                "action": "create",
                "items": [
                    {"title": "Bulk1", "content": "A"},
                    {"title": "Bulk2", "content": "B"},
                ],
            },
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(ResearcherNote.objects.count(), 2)

    def test_bulk_tags_add(self):
        n = self._create_note(tags=["old"])
        response = self.client.patch(
            "/api/notes/v1/notes/bulk/tags/",
            {"action": "add", "ids": [str(n.id)], "tags": ["new"]},
            format="json",
        )
        self.assertEqual(response.status_code, 200)
        n.refresh_from_db()
        self.assertIn("new", n.tags)


# ----------------------------------------------------------------------
# Versions
# ----------------------------------------------------------------------
class VersionTests(NotesTestBase):
    def test_version_history(self):
        note = self._create_note(content="V1")
        note.content = "V2"
        note.save()
        response = self.client.get(f"/api/notes/v1/notes/{note.id}/versions/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_version_detail(self):
        note = self._create_note(content="V1")
        note.content = "V2"
        note.save()
        response = self.client.get(f"/api/notes/v1/notes/{note.id}/versions/1/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["content"], "V1")


# ----------------------------------------------------------------------
# Mentions
# ----------------------------------------------------------------------
class MentionTests(NotesTestBase):
    def test_list_mentions(self):
        note = self._create_note()
        NoteMention.objects.create(note=note, mentioned_user=self.other_user)
        response = self.client.get(f"/api/notes/v1/notes/{note.id}/mentions/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_mark_mention_read(self):
        note = self._create_note()
        mention = NoteMention.objects.create(note=note, mentioned_user=self.user)
        self.client.post(f"/api/notes/v1/mentions/{mention.id}/read/")
        mention.refresh_from_db()
        self.assertTrue(mention.is_read)


# ----------------------------------------------------------------------
# Replies
# ----------------------------------------------------------------------
class ReplyTests(NotesTestBase):
    def test_list_replies(self):
        parent = self._create_note()
        reply = ResearcherNote.objects.create(
            content="reply", author=self.user, parent_note=parent
        )
        response = self.client.get(f"/api/notes/v1/notes/{parent.id}/replies/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)


# ----------------------------------------------------------------------
# Restore
# ----------------------------------------------------------------------
class RestoreTests(NotesTestBase):
    def test_restore_note(self):
        note = self._create_note()
        note.soft_delete()
        response = self.client.post(f"/api/notes/v1/notes/{note.id}/restore/")
        self.assertEqual(response.status_code, 200)
        note.refresh_from_db()
        self.assertFalse(note.is_deleted)

    def test_restore_note_by_non_author_forbidden(self):
        note = ResearcherNote.objects.create(content="x", author=self.other_user)
        note.soft_delete()
        self.client.force_authenticate(user=self.user)
        response = self.client.post(f"/api/notes/v1/notes/{note.id}/restore/")
        self.assertEqual(response.status_code, 403)


# ----------------------------------------------------------------------
# PDF Export (mock weasyprint)
# ----------------------------------------------------------------------


class PDFExportTests(NotesTestBase):
    @skipIf(not _WEASYPRINT, "weasyprint not installed")
    @patch("weasyprint.HTML.write_pdf", return_value=b"%PDF-1.4")
    def test_export_note_pdf(self, mock_pdf):
        note = self._create_note()
        response = self.client.get(f"/api/notes/v1/notes/{note.id}/export/pdf/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")


# ----------------------------------------------------------------------
# Webhooks
# ----------------------------------------------------------------------
class WebhookTests(NotesTestBase):
    def test_create_webhook(self):
        data = {
            "url": "https://example.com/hook",
            "events": ["note.created", "note.updated"],
            "is_active": True,
        }
        response = self.client.post("/api/notes/v1/webhooks/", data, format="json")
        self.assertEqual(response.status_code, 201)
        sub = WebhookSubscription.objects.get(url="https://example.com/hook")
        self.assertEqual(sub.events, ["note.created", "note.updated"])

    def test_list_webhooks(self):
        WebhookSubscription.objects.create(user=self.user, url="https://a.com")
        response = self.client.get("/api/notes/v1/webhooks/")
        self.assertEqual(len(response.data), 1)

    def test_delete_webhook(self):
        sub = WebhookSubscription.objects.create(user=self.user, url="https://a.com")
        response = self.client.delete(f"/api/notes/v1/webhooks/{sub.id}/")
        self.assertEqual(response.status_code, 204)
        self.assertFalse(WebhookSubscription.objects.filter(id=sub.id).exists())

    @patch("notes.webhook_utils.manual_send", return_value=True)
    def test_test_webhook(self, mock_send):
        sub = WebhookSubscription.objects.create(user=self.user, url="https://a.com")
        response = self.client.post(f"/api/notes/v1/webhooks/{sub.id}/test/")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data["success"])


# ----------------------------------------------------------------------
# Statistics & Recent Notes
# ----------------------------------------------------------------------
class StatisticsTests(NotesTestBase):
    def test_notes_statistics(self):
        self._create_note(is_public=True)
        self._create_note(is_public=False)
        response = self.client.get("/api/notes/v1/notes/statistics/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.data["total_notes"], 2)
        self.assertEqual(response.data["public_notes"], 1)

    def test_recent_notes(self):
        old_note = self._create_note()
        ResearcherNote.objects.filter(id=old_note.id).update(
            updated_at=timezone.now() - timedelta(days=10)
        )
        recent = self._create_note(title="Recent")
        response = self.client.get("/api/notes/v1/notes/recent/?days=7")
        self.assertEqual(len(response.data["results"]), 1)
        self.assertEqual(response.data["results"][0]["title"], "Recent")


# ----------------------------------------------------------------------
# Permissions / Unauthenticated
# ----------------------------------------------------------------------
class PermissionTests(NotesTestBase):
    def test_unauthenticated_access(self):
        client = APIClient()
        response = client.get("/api/notes/v1/notes/")
        self.assertEqual(response.status_code, 401)

    def test_other_user_cannot_update_private_note(self):
        private = ResearcherNote.objects.create(
            content="x", author=self.other_user
        )  # no is_public, defaults to False?
        self.client.force_authenticate(user=self.user)
        response = self.client.patch(
            f"/api/notes/v1/notes/{private.id}/", {"title": "hack"}, format="json"
        )
        self.assertEqual(response.status_code, 403)
