#!/usr/bin/env python3
"""
Kaggle API Credentials Configurator & Connection Tester
======================================================
Usage:
    python setup_kaggle_credentials.py --username <USERNAME> --key <API_KEY>
    python setup_kaggle_credentials.py --file /path/to/kaggle.json
    python setup_kaggle_credentials.py --status
"""

import os
import sys
import json
import argparse
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"
KAGGLE_FILE = KAGGLE_DIR / "kaggle.json"

def configure_credentials(username: str = None, key: str = None, json_file: str = None):
    """Save Kaggle credentials to ~/.kaggle/kaggle.json with 600 permissions."""
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)

    if json_file and os.path.exists(json_file):
        with open(json_file) as f:
            data = json.load(f)
        username = data.get("username")
        key = data.get("key")

    if not username or not key:
        print("❌ Error: Both username and API key are required.")
        return False

    creds = {"username": username, "key": key}
    with open(KAGGLE_FILE, "w") as f:
        json.dump(creds, f, indent=2)

    # Set strict 600 permissions as required by Kaggle CLI
    os.chmod(KAGGLE_FILE, 0o600)
    print(f"✓ Saved Kaggle API credentials to {KAGGLE_FILE} (chmod 600)")
    return test_connection()


def test_connection():
    """Test Kaggle API authentication."""
    print("🔄 Testing Kaggle API Connection...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print(f"✅ Kaggle API Authenticated Successfully!")
        # Try listing user datasets to confirm full API access
        datasets = api.dataset_list(user=api.username, page_size=3)
        print(f"   Logged in as: {api.username}")
        print(f"   Active Kaggle Datasets: {len(datasets)}")
        return True
    except Exception as e:
        print(f"❌ Kaggle API Authentication Failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Kaggle API Credentials Configurator")
    parser.add_argument("--username", help="Kaggle Username")
    parser.add_argument("--key", help="Kaggle API Key")
    parser.add_argument("--file", help="Path to kaggle.json file")
    parser.add_argument("--status", action="store_true", help="Test current connection status")

    args = parser.parse_args()

    if args.status:
        test_connection()
    elif args.username and args.key:
        configure_credentials(username=args.username, key=args.key)
    elif args.file:
        configure_credentials(json_file=args.file)
    else:
        if KAGGLE_FILE.exists():
            test_connection()
        else:
            print("ℹ Kaggle credentials not found at ~/.kaggle/kaggle.json")
            print("Please obtain your Kaggle API key from https://www.kaggle.com/settings -> 'Create New Token'")
            print("Then run:")
            print("  python setup_kaggle_credentials.py --username <YOUR_USERNAME> --key <YOUR_API_KEY>")


if __name__ == "__main__":
    main()
