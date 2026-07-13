#!/bin/zsh

# Shared helpers for the repository's endpoint diagnostic scripts.

typeset -ga TEMP_FILES=()

cleanup_temp_files() {
  local temp_path
  for temp_path in "${TEMP_FILES[@]}"; do
    [[ -n "$temp_path" ]] && rm -f -- "$temp_path"
  done
}

trap cleanup_temp_files EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

make_temp_file() {
  REPLY="$(mktemp)"
  TEMP_FILES+=("$REPLY")
}

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
  make_temp_file
  body="$REPLY"

  if ! http_status="$(curl -sS -m 5 -o "$body" -w '%{http_code}' "$ngrok_api")"; then
    return 1
  fi
  if [[ "$http_status" != "200" ]]; then
    return 1
  fi

  local public_url
  local expected_port="${LLM_PROXY_PORT:-8330}"
  public_url="$(jq -r --arg suffix ":${expected_port}" '
    [
      .tunnels[]?
      | select(.public_url | startswith("https://"))
      | select((.config.addr // "") | endswith($suffix))
      | .public_url
    ][0] // empty
  ' "$body")"

  if [[ -z "$public_url" ]]; then
    local -a candidates
    candidates=("${(@f)$(jq -r '.tunnels[]? | select(.public_url | startswith("https://")) | .public_url' "$body")}")
    if (( ${#candidates[@]} != 1 )); then
      return 1
    fi
    public_url="${candidates[1]}"
  fi

  normalize_v1_base "$public_url"
}
