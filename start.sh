#!/usr/bin/env bash
set -e

# Render injeta PORT automaticamente
echo "Starting server on port: ${PORT}"

exec uvicorn app:app --host 0.0.0.0 --port "${PORT}" --proxy-headers --forwarded-allow-ips="*"
