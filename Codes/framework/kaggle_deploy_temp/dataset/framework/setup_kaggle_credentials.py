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
import subprocess
from pathlib import Path

KAGGLE_DIR = Path.home() / ".kaggle"
TOKEN_FILE = KAGGLE_DIR / "access_token"

def configure_token(token: str):
    """Save Kaggle Access Token (KGAT_...) to ~/.kaggle/access_token with 600 permissions."""
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TOKEN_FILE, "w") as f:
        f.write(token.strip())
    os.chmod(TOKEN_FILE, 0o600)
    os.environ["KAGGLE_API_TOKEN"] = token.strip()
    print(f"✓ Saved Kaggle Access Token to {TOKEN_FILE} (chmod 600)")
    return test_connection()

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
        # Query datasets via API
        res = subprocess.run(["kaggle", "datasets", "list", "--mine"], capture_output=True, text=True)
        if res.returncode == 0:
            print("   API CLI status: Connected & operational")
            lines = res.stdout.strip().splitlines()
            if len(lines) > 2:
                print(f"   Sample user dataset: {lines[2].split()[0]}")
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
