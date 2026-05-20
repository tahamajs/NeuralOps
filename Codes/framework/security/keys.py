"""
Key generation, serialization, and loading utilities.
Uses the `cryptography` library for strong, standardised operations.
"""

import os
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization

def generate_ed25519_keypair(save_to: str = None) -> tuple:
    """
    Generate a fresh Ed25519 key pair.

    Args:
        save_to: if a directory is provided, the private key is saved as
                 ``ed25519_private.pem`` and the public key as ``ed25519_public.pem``
                 inside that directory.

    Returns:
        (private_key, public_key) :py:class:`cryptography` objects
    """
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()

    if save_to:
        out = Path(save_to)
        out.mkdir(parents=True, exist_ok=True)

        # Private key – keep it secret!
        priv_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        (out / "ed25519_private.pem").write_bytes(priv_bytes)

        # Public key
        pub_bytes = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        (out / "ed25519_public.pem").write_bytes(pub_bytes)

    return private_key, public_key


def load_private_key(path: str) -> ed25519.Ed25519PrivateKey:
    """Load an Ed25519 private key from a PEM file."""
    pem = Path(path).read_bytes()
    return serialization.load_pem_private_key(pem, password=None)


def load_public_key(path: str) -> ed25519.Ed25519PublicKey:
    """Load an Ed25519 public key from a PEM file."""
    pem = Path(path).read_bytes()
    return serialization.load_pem_public_key(pem)