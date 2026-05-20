"""
Model artifact signing and verification.

Signs a model file using an Ed25519 private key and stores the signature
in a ``.sig`` file next to the model. Verification checks the signature
against the corresponding public key.
"""

import hashlib
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature

from .hashes import compute_file_hash

def sign_model(model_file: str, private_key: ed25519.Ed25519PrivateKey) -> None:
    """
    Sign a model artifact.

    The signature is stored in ``<model_file>.sig`` as a base64‑encoded string.

    Args:
        model_file: path to the model file (e.g., ``pytorch_model.bin``).
        private_key: an Ed25519 private key object.
    """
    digest = compute_file_hash(model_file, "sha256")
    signature_bytes = private_key.sign(digest.encode("utf-8"))
    signature_str = signature_bytes.hex()   # safe for file storage

    sig_path = model_file + ".sig"
    Path(sig_path).write_text(signature_str)


def verify_signature(model_file: str,
                     public_key: ed25519.Ed25519PublicKey) -> bool:
    """
    Verify the signature of a model artifact.

    Args:
        model_file: path to the model file.
        public_key: an Ed25519 public key object.

    Returns:
        ``True`` if the signature is valid, otherwise ``False``.
    """
    sig_path = model_file + ".sig"
    sig_file = Path(sig_path)
    if not sig_file.exists():
        raise FileNotFoundError(f"Signature file not found: {sig_path}")

    digest = compute_file_hash(model_file, "sha256")
    signature_str = sig_file.read_text().strip()
    signature_bytes = bytes.fromhex(signature_str)

    try:
        public_key.verify(signature_bytes, digest.encode("utf-8"))
        return True
    except InvalidSignature:
        return False


def generate_keypair(save_to: str = None) -> tuple:
    """
    Convenience wrapper around :func:`framework.security.keys.generate_ed25519_keypair`.
    """
    from .keys import generate_ed25519_keypair
    return generate_ed25519_keypair(save_to)