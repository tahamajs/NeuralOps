# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, please **DO NOT** open a public issue.

Instead, email **tahamaj4@gmail.com** with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact

We will respond within 48 hours.

## Supported Versions

| Version | Supported |
|:---|:---:|
| 1.2.x | ✅ |
| 1.1.x | ✅ |
| 1.0.x | ⚠️ Security only |
| < 1.0 | ❌ |

## Security Best Practices

When using NeuralOps:
- Never load models from untrusted sources
- Use `safetensors` format instead of pickle
- Verify model checksums
- Run in isolated environments for untrusted models
