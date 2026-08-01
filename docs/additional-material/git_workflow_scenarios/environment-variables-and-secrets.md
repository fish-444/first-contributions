# Environment Variables and Secrets

Almost every real-world project needs some values that should **not** live in the code itself — database passwords, API keys, access tokens, and other secrets. This document explains what environment variables are, why secrets must be kept out of Git, and how to handle them safely.

**Note:** Always follow the security policies of your place of work/study.

## What is an Environment Variable?

An environment variable is a named value that lives *outside* your source code, in the environment where your program runs. Your program reads it at runtime instead of having the value hard-coded.

- **Node.js:** `process.env.API_KEY`
- **Python:** `os.environ["API_KEY"]`
- **Shell:** `echo $API_KEY`

Because the value is supplied by the environment, the same code can run with different settings in development, testing, and production — without changing a single line.

## Why Keep Secrets Out of Git?

Once a secret is committed and pushed, it is effectively **public forever**, even if you delete it later:

- Anyone who can read the repository can read the secret.
- Git history keeps every past version, so removing the line in a new commit is **not** enough.
- Forks, clones, and caches may still contain the old value.
- Bots continuously scan public repositories for leaked keys, often within seconds of a push.

The safest rule is simple: **never commit a secret in the first place.**

## Using a `.env` File

A common pattern is to store secrets in a local `.env` file that is **never** committed:

```sh
# .env
API_KEY=sk_live_1234567890
DATABASE_URL=postgres://user:password@localhost:5432/mydb
```

Add the file to your [`.gitignore`](creating-a-gitignore-file.md) so Git never tracks it:

```sh
# .gitignore
.env
.env.*
!.env.example
```

Then load it in your application. Many languages have a helper library for this, for example [`dotenv`](https://www.npmjs.com/package/dotenv) for Node.js or [`python-dotenv`](https://pypi.org/project/python-dotenv/) for Python.

## Sharing the *Shape* with `.env.example`

Teammates still need to know **which** variables your project expects. Commit a template file with the keys but **no real values**:

```sh
# .env.example
API_KEY=
DATABASE_URL=
```

Now anyone can copy it and fill in their own secrets:

```sh
cp .env.example .env
```

The `!.env.example` line in the `.gitignore` above makes sure this template is *not* ignored, even though all other `.env.*` files are.

## Secrets in GitHub Actions and CI

For automated workflows, don't put secrets in your workflow files. Store them as **encrypted secrets** in your repository settings instead:

1. Go to **Settings → Secrets and variables → Actions**.
2. Click **New repository secret** and add your key and value.
3. Reference it in a workflow with the `secrets` context:

```yaml
steps:
  - name: Deploy
    env:
      API_KEY: ${{ secrets.API_KEY }}
    run: ./deploy.sh
```

GitHub keeps these values encrypted and automatically masks them in logs.

## Help! I Already Committed a Secret

If a secret slips into a commit, treat it as compromised:

1. **Rotate it immediately.** Revoke the leaked key or password and generate a new one — this is the most important step.
2. **Remove it from history.** A single commit can be cleaned with `git rm --cached`, but if it was pushed you'll need to rewrite history with a tool such as [`git filter-repo`](https://github.com/newren/git-filter-repo) or the [BFG Repo-Cleaner](https://rtyley.github.io/bfg-repo-cleaner/).
3. **Force-push and notify collaborators**, since rewriting history changes commit hashes.

Removing the file without rotating the secret is **not** enough — assume anything that was pushed has already been seen.

## Summary

- Keep secrets in environment variables, not in code.
- Store them locally in a `.gitignore`d `.env` file.
- Commit a `.env.example` template so others know what's needed.
- Use encrypted secrets for CI/CD instead of plaintext.
- If you leak a secret, **rotate it first**, then clean history.

### [Additional Material](additional-material.md)
