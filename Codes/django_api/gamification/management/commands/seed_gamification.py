from django.core.management.base import BaseCommand
from gamification.models import Badge, PremiumFeature


class Command(BaseCommand):
    help = 'Seed default badges and premium features'

    def handle(self, *args, **options):
        badges = [
            ('First Paper', 'Save your first research paper', '📄', 'bronze', 'publication', {'papers_saved': 1}),
            ('Collector', 'Save 10 research papers', '📚', 'silver', 'publication', {'papers_saved': 10}),
            ('Librarian', 'Save 50 research papers', '📖', 'gold', 'publication', {'papers_saved': 50}),
            ('First Note', 'Create your first research note', '📝', 'bronze', 'contribution', {'notes_created': 1}),
            ('Notetaker', 'Create 10 research notes', '🗒️', 'silver', 'contribution', {'notes_created': 10}),
            ('Reviewer', 'Complete 5 peer reviews', '✅', 'silver', 'contribution', {'reviews': 5}),
            ('7-Day Streak', 'Maintain a 7-day activity streak', '🔥', 'silver', 'streak', {'days_streak': 7}),
            ('30-Day Streak', 'Maintain a 30-day activity streak', '💎', 'gold', 'streak', {'days_streak': 30}),
            ('Centurion', 'Earn 100 total points', '💯', 'silver', 'milestone', {'total_points': 100}),
            ('Grandmaster', 'Earn 1000 total points', '👑', 'gold', 'milestone', {'total_points': 1000}),
            ('Legend', 'Earn 10000 total points', '🌟', 'platinum', 'milestone', {'total_points': 10000}),
            ('Helpful', 'Have 5 answers accepted', '💡', 'silver', 'social', {'answers_accepted': 5}),
        ]

        for name, desc, icon, tier, cat, criteria in badges:
            Badge.objects.get_or_create(
                name=name,
                defaults={
                    'description': desc,
                    'icon': icon,
                    'tier': tier,
                    'category': cat,
                    'criteria': criteria,
                }
            )

        features = [
            ('Extended File Storage', 'Upload larger files to experiments', 500),
            ('Priority Job Scheduling', 'Jump to the front of the job queue', 1000),
            ('Advanced Analytics', 'Access advanced experiment analytics dashboard', 2000),
            ('API Access', 'Access the NeuralOps REST API programmatically', 5000),
            ('Custom Badge', 'Create your own custom profile badge', 10000),
        ]

        for name, desc, points in features:
            PremiumFeature.objects.get_or_create(
                name=name,
                defaults={
                    'description': desc,
                    'required_points': points,
                }
            )

        self.stdout.write(self.style.SUCCESS(f'Seeded {len(badges)} badges and {len(features)} premium features'))