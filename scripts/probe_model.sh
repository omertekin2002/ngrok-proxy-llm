#!/bin/zsh

set -euo pipefail

if [[ $# -lt 1 ]]; then
  print "Usage: ./scripts/probe_model.sh MODEL [PROMPT]" >&2
  exit 1
fi

MODEL_ID="$1"
PROMPT_TEXT="${2:-Reply with exactly ok}"

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

summarize_body() {
  local body="$1"
  if ! jq -e '.' "$body" >/dev/null 2>&1; then
    print "response:"
    head -c 400 "$body"
    print
    return 0
  fi

  local summary
  summary="$(jq -r '
    if .error then
      [
        "error.code: " + (.error.code // ""),
        "error.type: " + (.error.type // ""),
        "error.message: " + (.error.message // "")
      ] | join("\n")
    elif .choices and (.choices | length) > 0 then
      [
        "model: " + (.model // ""),
        "content: " + (((.choices[0].message.content // "") | tostring)[:200]),
        "images: " + (((.choices[0].message.images // []) | length) | tostring)
      ] | join("\n")
    else
      "top_keys: " + ((keys | join(", ")) // "")
    end
  ' "$body")"
  print "$summary"
}

probe_endpoint() {
  local name="$1"
  local base="$2"
  local body
  local payload
  local http_status

  body="$(mktemp)"
  payload="$(mktemp)"

  jq -n \
    --arg model "$MODEL_ID" \
    --arg prompt "$PROMPT_TEXT" \
    '{model: $model, messages: [{role: "user", content: $prompt}]}' > "$payload"

  http_status="$(curl -sS -m "${CURL_MAX_TIME:-180}" -o "$body" -w '%{http_code}' \
    "${base%/}/chat/completions" \
    -H 'Content-Type: application/json' \
    --data @"$payload" || true)"

  print
  print "[$name] ${base%/}/chat/completions"
  print "status: $http_status"
  summarize_body "$body"

  rm -f "$body" "$payload"
}

local_base="$(normalize_v1_base "${LLM_LOCAL_URL:-http://localhost:8317}")"
proxy_base="$(normalize_v1_base "${LLM_PROXY_URL:-http://localhost:8330}")"

print "Model probe"
print "model: $MODEL_ID"
print "prompt: $PROMPT_TEXT"

probe_endpoint "local" "$local_base"
probe_endpoint "proxy" "$proxy_base"

if public_base="$(detect_public_base 2>/dev/null)"; then
  probe_endpoint "public" "$public_base"
else
  print
  print "[public] skipped"
  print "PUBLIC_BASE_URL is unset and ngrok admin API is unavailable."
fi
