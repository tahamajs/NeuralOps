import hashlib


def sign_model(model_file, private_key):
    """Create a SHA256 signature file next to a model artifact."""
    with open(model_file, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    signature = private_key.sign(digest.encode())
    if isinstance(signature, bytes):
        signature = signature.decode()
    with open(model_file + ".sig", 'w') as f:
        f.write(signature)


def verify_signature(model_file, public_key):
    """Verify the SHA256 signature file for a model artifact."""
    with open(model_file, 'rb') as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    with open(model_file + ".sig", 'r') as f:
        signature = f.read()
    return public_key.verify(digest.encode(), signature)
