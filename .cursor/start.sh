#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PATH="${HOME}/.local/bin:${PATH}"

if [[ ! -f .env ]]; then
  cp .env.example .env
  # Heuristic-only default for Cloud Agents (no local Ollama required).
  if grep -q '^LLM_PROVIDER=' .env; then
    sed -i 's/^LLM_PROVIDER=.*/LLM_PROVIDER=none/' .env
  else
    echo 'LLM_PROVIDER=none' >> .env
  fi
fi
