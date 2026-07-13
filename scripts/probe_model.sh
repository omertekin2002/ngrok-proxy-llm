#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 1 ]]; then
  print "Usage: ./scripts/probe_model.sh MODEL [PROMPT]" >&2
  exit 1
fi

MODEL_ID="$1"
PROMPT_TEXT="${2:-Reply with exactly ok}"

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

  make_temp_file
  body="$REPLY"
  make_temp_file
  payload="$REPLY"

  jq -n \
    --arg model "$MODEL_ID" \
    --arg prompt "$PROMPT_TEXT" \
    '{model: $model, messages: [{role: "user", content: $prompt}]}' > "$payload"

  if ! http_status="$(curl -sS -m "${CURL_MAX_TIME:-180}" -o "$body" -w '%{http_code}' \
    "${base%/}/chat/completions" \
    -H 'Content-Type: application/json' \
    --data @"$payload")"; then
    http_status="${http_status:-000}"
  fi

  print
  print "[$name] ${base%/}/chat/completions"
  print "status: $http_status"
  summarize_body "$body"
  [[ "$http_status" == "200" ]]
}

local_base="$(normalize_v1_base "${LLM_LOCAL_URL:-http://localhost:8317}")"
proxy_base="$(normalize_v1_base "${LLM_PROXY_URL:-http://localhost:8330}")"

print "Model probe"
print "model: $MODEL_ID"
print "prompt: $PROMPT_TEXT"

failures=0
probe_endpoint "local" "$local_base" || (( failures += 1 ))
probe_endpoint "proxy" "$proxy_base" || (( failures += 1 ))

if public_base="$(detect_public_base 2>/dev/null)"; then
  probe_endpoint "public" "$public_base" || (( failures += 1 ))
else
  print
  print "[public] skipped"
  print "PUBLIC_BASE_URL is unset and ngrok admin API is unavailable."
fi

(( failures == 0 ))
