# framework/security/__init__.py
from .signing import sign_model, verify_signature

# framework/security/signing.py
import hashlib
import json

def sign_model(model_file, private_key):
    """Add a SHA256 signature to the model file."""
    with open(model_file, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    signature = private_key.sign(digest.encode())
    with open(model_file + ".sig", 'w') as f:
        f.write(signature)

def verify_signature(model_file, public_key):
    """Verify the signature."""
    with open(model_file, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(model_file + ".sig", 'r') as f:
        signature = f.read()
    return public_key.verify(digest.encode(), signature)