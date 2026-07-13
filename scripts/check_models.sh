#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
source "$SCRIPT_DIR/common.sh"

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

  make_temp_file
  body="$REPLY"
  if ! http_status="$(curl -sS -m "${CURL_MAX_TIME:-30}" -o "$body" -w '%{http_code}' "$url")"; then
    http_status="${http_status:-000}"
  fi

  print
  print "[$name] $url"
  print "status: $http_status"

  if [[ "$http_status" == "200" ]] && jq -e '.data' "$body" >/dev/null 2>&1; then
    print_models "$body"
    return 0
  else
    print "response:"
    head -c 400 "$body"
    print
  fi

  return 1
}

local_base="$(normalize_v1_base "${LLM_LOCAL_URL:-http://localhost:8317}")"
proxy_base="$(normalize_v1_base "${LLM_PROXY_URL:-http://localhost:8330}")"

print "Model availability check"

failures=0
check_endpoint "local" "$local_base" || (( failures += 1 ))
check_endpoint "proxy" "$proxy_base" || (( failures += 1 ))

if public_base="$(detect_public_base 2>/dev/null)"; then
  check_endpoint "public" "$public_base" || (( failures += 1 ))
else
  print
  print "[public] skipped"
  print "PUBLIC_BASE_URL is unset and ngrok admin API is unavailable."
fi

(( failures == 0 ))
