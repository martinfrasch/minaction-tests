# Security Best Practices

This document outlines security best practices for working with this repository, particularly regarding API keys and credentials.

## API Key Management

### ✅ DO

**Use Environment Variables**
```bash
# Set API keys in your shell environment
export GEMINI_API_KEY="your-actual-key-here"
export ANTHROPIC_API_KEY="your-actual-key-here"
```

**Use .env Files (Locally Only)**
```bash
# Create a .env file (already in .gitignore)
echo "GEMINI_API_KEY=your-actual-key-here" > .env
echo "ANTHROPIC_API_KEY=your-actual-key-here" >> .env

# Load with python-dotenv
python -c "from dotenv import load_dotenv; load_dotenv()"
```

**Use Secret Managers (Production)**
- Google Cloud Secret Manager
- AWS Secrets Manager
- HashiCorp Vault
- Azure Key Vault

### ❌ DON'T

**Never Hardcode API Keys**
```python
# ❌ BAD - Never do this!
api_key = "AIzaSyD-9tNaCJI..."
```

**Never Commit Credentials**
- Don't commit `.env` files
- Don't commit `credentials.json` files
- Don't commit private keys (`.pem`, `.key` files)

**Never Share Keys in Documentation**
```markdown
❌ BAD:
export GEMINI_API_KEY="AIzaSyD-9tNaCJI..."

✅ GOOD:
export GEMINI_API_KEY="your-key-here"
```

## Protected Files

The `.gitignore` file protects the following:

### Credentials
- `.env`, `.env.*`
- `*.key`, `*.pem`, `*.p12`, `*.pfx`
- `credentials.json`, `*-credentials.json`
- `service-account*.json`
- `api-keys.txt`, `secrets.txt`

### Sensitive Outputs
- `results/batch/` (may contain API responses)
- `*.log` (may contain keys in error messages)

## Checking for Leaked Credentials

### Before Committing

```bash
# Check for potential API keys in staged files
git diff --cached | grep -E "AIza[0-9A-Za-z_-]{35}|sk-[a-zA-Z0-9]{48}"

# Check all files for patterns
grep -r -E "api[_-]?key.*=.*['\"][a-zA-Z0-9]{20,}" . --include="*.py"
```

### Scan Repository History

```bash
# Use git-secrets (install first)
git secrets --scan

# Or manually check history
git log -p | grep -E "AIza[0-9A-Za-z_-]{35}|sk-[a-zA-Z0-9]{48}"
```

## If You Accidentally Commit a Key

### 1. Revoke the Key Immediately

**Google Cloud API Keys:**
1. Go to [Google Cloud Console](https://console.cloud.google.com/apis/credentials)
2. Find the compromised key
3. Click "Delete" or "Regenerate"

**Anthropic API Keys:**
1. Go to [Anthropic Console](https://console.anthropic.com/settings/keys)
2. Revoke the compromised key
3. Generate a new one

### 2. Remove from Git History

```bash
# Using git-filter-repo (recommended)
git filter-repo --path-match 'file-with-key' --invert-paths

# Or use BFG Repo-Cleaner
bfg --replace-text secrets.txt
```

### 3. Force Push (Caution!)

```bash
# Only if you're sure no one else is using the branch
git push --force origin <branch-name>
```

### 4. Rotate All Keys

Even if you removed the key from history, assume it was compromised. Generate new keys for all services.

## API Key Scoping

### Google Cloud API Keys

Restrict your API keys:
1. **API Restrictions**: Enable only Generative Language API
2. **Application Restrictions**:
   - Set HTTP referrers for web apps
   - Set IP addresses for backend apps
3. **Quota Limits**: Set reasonable daily limits

### Best Practices

- Use separate keys for development and production
- Set expiration dates where possible
- Monitor API usage for anomalies
- Use service accounts with minimal permissions

## Environment-Specific Configuration

### Development
```bash
# ~/.bashrc or ~/.zshrc
export GEMINI_API_KEY="dev-key-here"
```

### CI/CD
```yaml
# GitHub Actions
env:
  GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
```

### Docker
```bash
# Pass as build arg (not recommended for sensitive data)
docker build --build-arg API_KEY=$GEMINI_API_KEY .

# Better: Use secrets
docker run --env-file .env.production myimage
```

## Security Checklist

Before committing:
- [ ] No hardcoded API keys in code
- [ ] No keys in configuration files
- [ ] `.env` files are in `.gitignore`
- [ ] No keys in commit messages
- [ ] No keys in documentation examples
- [ ] Test credentials are clearly marked as examples

Before deploying:
- [ ] Production keys are in secret manager
- [ ] API keys have appropriate restrictions
- [ ] Monitoring is set up for API usage
- [ ] Keys are rotated regularly
- [ ] Access logs are enabled

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do NOT** open a public issue
2. Email the maintainer directly: mfrasch@uw.edu
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Additional Resources

- [Google Cloud API Security Best Practices](https://cloud.google.com/docs/security/best-practices)
- [OWASP API Security Top 10](https://owasp.org/www-project-api-security/)
- [GitHub Secret Scanning](https://docs.github.com/en/code-security/secret-scanning)
- [git-secrets](https://github.com/awslabs/git-secrets)
- [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/)

## License

This security documentation is provided under the same MIT License as the repository.
