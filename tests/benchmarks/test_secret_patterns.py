from neuralops.benchmarks.results.hf_sync import SECRET_PATTERNS


def test_expanded_secret_patterns():
    # 8 patterns verified
    assert len(SECRET_PATTERNS) == 8

    samples = [
        "hf_abcdefghijklmnopqrstuvwxyz1234567890",  # HF
        "sk-abcdefghijklmnopqrstuvwxyz1234567890",  # OpenAI
        "ghp_123456789012345678901234567890123456",  # GitHub personal
        "ghs_123456789012345678901234567890123456",  # GitHub server
        "AKIAIOSFODNN7EXAMPLE",  # AWS
        "AIzaSyD-xyz1234567890123456789012345678",  # Google (AIza + 35 chars)
        "pypi-AgEIcHlwaS5vcmc-example-token-that-is-at-least-fifty-chars-long-and-more",
        'api_key="1234567890abcdef123"',
    ]

    for sample in samples:
        matched = any(p.search(sample) for p in SECRET_PATTERNS)
        assert matched, f"Failed to detect secret sample: {sample}"
