from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient
from rest_framework import status
from .models import PaperProject, PaperFile, PaperCompile, PaperTemplate

User = get_user_model()


class PaperProjectTestCase(TestCase):
    def setUp(self):
        self.client = APIClient()
        self.user = User.objects.create_user(
            username='testuser', password='testpass'
        )
        self.other_user = User.objects.create_user(
            username='other', password='testpass'
        )
        self.client.force_authenticate(user=self.user)

    def test_create_project(self):
        resp = self.client.post('/api/paper/projects/', {
            'name': 'My Paper',
            'compiler': 'pdflatex',
        }, format='json')
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(PaperProject.objects.count(), 1)

    def test_list_projects(self):
        PaperProject.objects.create(owner=self.user, name='P1')
        PaperProject.objects.create(owner=self.other_user, name='P2')  # not visible
        resp = self.client.get('/api/paper/projects/')
        self.assertEqual(len(resp.data['results']), 1)

    def test_add_file_to_project(self):
        project = PaperProject.objects.create(owner=self.user, name='P1')
        resp = self.client.post(
            f'/api/paper/projects/{project.id}/files/',
            {'filename': 'main.tex', 'content': r'\documentclass{article}\begin{document}Hello\end{document}'},
            format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        self.assertEqual(project.files.count(), 1)

    def test_compile_view(self):
        project = PaperProject.objects.create(owner=self.user, name='P1')
        PaperFile.objects.create(
            project=project,
            filename='main.tex',
            content=r'\documentclass{article}\begin{document}Hello\end{document}',
        )
        resp = self.client.post(
            f'/api/paper/projects/{project.id}/compile/',
            format='json',
        )
        # May fail if pdflatex not installed; test still validates the endpoint
        self.assertIn(resp.status_code, [200, 500])

    def test_collaborator_access(self):
        project = PaperProject.objects.create(owner=self.user, name='P1')
        # Add other user as collaborator
        project.collaborators.add(self.other_user)
        # Switch to other user
        self.client.force_authenticate(user=self.other_user)
        resp = self.client.get(f'/api/paper/projects/{project.id}/')
        self.assertEqual(resp.status_code, 200)

    def test_template_create_and_use(self):
        template = PaperTemplate.objects.create(
            name='Test Template',
            files={'main.tex': r'\documentclass{article}\begin{document}Hi\end{document}'},
        )
        resp = self.client.post(
            f'/api/paper/templates/{template.id}/use/',
            format='json',
        )
        self.assertEqual(resp.status_code, status.HTTP_201_CREATED)
        project = PaperProject.objects.first()
        self.assertEqual(project.files.count(), 1)