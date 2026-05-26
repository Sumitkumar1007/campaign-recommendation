# AGENTS.md

- Read `docs/codex_handoff.md` before starting project work.
- Be token-efficient: use targeted `rg`/file reads, not full repo scans, unless necessary.
- GitHub only; do not push to or operate on GitLab unless explicitly asked.
- Use `./venv/bin/python` for scripts and tests.
- For MCollect export, dry-run first. Use `--write` only when explicitly requested.
- Do not commit `.env`, raw data, model binaries, generated predictions, logs, or large artifacts.
