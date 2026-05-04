def manual_send(url, payload, secret):
    """Send a webhook payload synchronously (for testing)."""
    import requests, hmac, hashlib, json
    headers = {'Content-Type': 'application/json'}
    if secret:
        signature = hmac.new(secret.encode(), json.dumps(payload).encode(), hashlib.sha256).hexdigest()
        headers['X-Signature'] = signature
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=5)
        return resp.status_code < 400
    except Exception:
        return False