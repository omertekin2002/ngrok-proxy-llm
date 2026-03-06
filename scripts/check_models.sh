#!/bin/zsh

set -euo pipefail

normalize_v1_base() {
  local base="${1%/}"
  if [[ -z "$base" ]]; then
    return 1
  fi

  if [[ "$base" == */v1 ]]; then
    print -r -- "$base"
  else
    print -r -- "$base/v1"
  fi
}

detect_public_base() {
  local explicit="${PUBLIC_BASE_URL:-}"
  if [[ -n "$explicit" ]]; then
    normalize_v1_base "$explicit"
    return 0
  fi

  local ngrok_api="${NGROK_API_URL:-http://127.0.0.1:4040/api/tunnels}"
  local body
  local http_status
  body="$(mktemp)"
  http_status="$(curl -sS -m 5 -o "$body" -w '%{http_code}' "$ngrok_api" || true)"

  if [[ "$http_status" != "200" ]]; then
    rm -f "$body"
    return 1
  fi

  local public_url
  public_url="$(jq -r '.tunnels[]? | select(.public_url | startswith("https://")) | .public_url' "$body" | head -n 1)"
  rm -f "$body"

  if [[ -z "$public_url" ]]; then
    return 1
  fi

  normalize_v1_base "$public_url"
}

print_models() {
  local body="$1"
  jq -r '.data[]?.id' "$body" | sort
}

check_endpoint() {
  local name="$1"
  local base="$2"
  local url="${base%/}/models"
  local body
  local http_status

  body="$(mktemp)"
  http_status="$(curl -sS -m "${CURL_MAX_TIME:-30}" -o "$body" -w '%{http_code}' "$url" || true)"

  print
  print "[$name] $url"
  print "status: $http_status"

  if [[ "$http_status" == "200" ]] && jq -e '.data' "$body" >/dev/null 2>&1; then
    print_models "$body"
  else
    print "response:"
    head -c 400 "$body"
    print
  fi

  rm -f "$body"
}

local_base="$(normalize_v1_base "${LLM_LOCAL_URL:-http://localhost:8317}")"
proxy_base="$(normalize_v1_base "${LLM_PROXY_URL:-http://localhost:8330}")"

print "Model availability check"

check_endpoint "local" "$local_base"
check_endpoint "proxy" "$proxy_base"

if public_base="$(detect_public_base 2>/dev/null)"; then
  check_endpoint "public" "$public_base"
else
  print
  print "[public] skipped"
  print "PUBLIC_BASE_URL is unset and ngrok admin API is unavailable."
fi
