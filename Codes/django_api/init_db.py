#!/usr/bin/env python
"""
Database Initialization Script
Initializes Django database with migrations and creates a superuser
"""
import os
import sys
import django
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django_api.settings')
sys.path.insert(0, str(Path(__file__).parent))

django.setup()

from django.core.management import call_command
from django.contrib.auth.models import User
from rest_framework.authtoken.models import Token

def init_database():
    """Initialize database with migrations and test data"""
    print("\n" + "="*80)
    print("DATABASE INITIALIZATION".center(80))
    print("="*80 + "\n")

    # Run migrations
    print("[1] Running migrations...")
    try:
        call_command('migrate')
        print("✓ Migrations completed successfully\n")
    except Exception as e:
        print(f"✗ Migration failed: {e}\n")
        return False

    # Create superuser if it doesn't exist
    print("[2] Creating superuser...")
    try:
        if not User.objects.filter(username='admin').exists():
            admin_user = User.objects.create_superuser(
                username='admin',
                email='admin@localhost.local',
                password='admin123'
            )
            Token.objects.get_or_create(user=admin_user)
            print("✓ Superuser 'admin' created")
            print("  Username: admin")
            print("  Email: admin@localhost.local")
            print("  Password: admin123\n")
        else:
            print("✓ Superuser 'admin' already exists\n")
    except Exception as e:
        print(f"✗ Superuser creation failed: {e}\n")
        return False

    # Create test users
    print("[3] Creating test users...")
    test_users = [
        {'username': 'researcher1', 'email': 'researcher1@test.com', 'password': 'researcher123'},
        {'username': 'researcher2', 'email': 'researcher2@test.com', 'password': 'researcher123'},
    ]
    
    for user_data in test_users:
        try:
            if not User.objects.filter(username=user_data['username']).exists():
                user = User.objects.create_user(**user_data)
                Token.objects.get_or_create(user=user)
                print(f"✓ User '{user_data['username']}' created")
            else:
                print(f"✓ User '{user_data['username']}' already exists")
        except Exception as e:
            print(f"✗ Failed to create user {user_data['username']}: {e}")
    
    print("\n[4] Database Status:")
    print(f"✓ Total users: {User.objects.count()}")
    print(f"✓ Database initialized successfully\n")

    print("="*80)
    print("INITIALIZATION COMPLETE".center(80))
    print("="*80 + "\n")
    
    return True

if __name__ == '__main__':
    try:
        success = init_database()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nInitialization interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nInitialization failed: {e}")
        sys.exit(1)
