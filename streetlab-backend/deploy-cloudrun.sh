#!/usr/bin/env bash
# Deploy the hosted backend to Google Cloud Run (replaces the old fly.toml).
# Usage:  PROJECT_ID=<gcp-project> ALLOWED_ORIGINS="https://a.example,https://b-*.vercel.app" ./deploy-cloudrun.sh
# Runbook, sizing and trade-offs: docs/hosted-sessions.md
set -euo pipefail
cd "$(dirname "$0")"

: "${PROJECT_ID:?set PROJECT_ID to your GCP project id}"
: "${ALLOWED_ORIGINS:?set ALLOWED_ORIGINS to the browser origins allowed to open the WebSocket}"

# --max-instances 1 is mandatory: sessions and the STREETLAB_MAX_SESSIONS cap live in one
#   process's memory, so a second instance would not know about the first one's sessions.
# --min-instances 0: scale to zero when idle (free tier). First visitor after idle pays a cold start.
# --timeout 3600: Cloud Run's maximum; it drops the WebSocket then and the frontend reconnects
#   with a fresh session.
# CPU throttling is left at the default (request-based billing): an open WebSocket counts as an
#   active request, so a live session gets full CPU and an idle service costs nothing.
# --source builds in Cloud Build, so an Apple-silicon Mac does not push an arm64 image.
# "^@^" makes @ the env-var separator, because ALLOWED_ORIGINS itself contains commas.
gcloud run deploy streetlab-sim \
  --source . \
  --project "$PROJECT_ID" \
  --region us-east1 \
  --port 8080 \
  --cpu 1 \
  --memory 1Gi \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 3600 \
  --allow-unauthenticated \
  --set-env-vars "^@^STREETLAB_ALLOWED_ORIGINS=${ALLOWED_ORIGINS}@STREETLAB_MAX_SESSIONS=2"
