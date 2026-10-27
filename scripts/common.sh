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

# Repository-root .env, located from this file so the scripts work from any cwd.
DIAGNOSTIC_ENV_FILE="${${(%):-%x}:A:h:h}/.env"

# Settings the diagnostic scripts read from the environment or .env.
typeset -ga DIAGNOSTIC_SETTINGS=(
  LLM_API_KEY
  LLM_LOCAL_URL
  LLM_PROXY_URL
  LLM_PROXY_PORT
  PUBLIC_BASE_URL
  NGROK_API_URL
  CURL_MAX_TIME
)

# Read the named keys from a KEY=value file without sourcing or executing it.
# Comments, blank lines and an optional `export ` prefix are ignored, matching
# quotes around a value are stripped, and the last assignment of a key wins.
# Variables already set in the environment are left alone.
load_env_file_settings() {
  local env_file="$1"
  shift
  [[ -f "$env_file" && -r "$env_file" ]] || return 0
  setopt local_options no_extended_glob no_glob_subst

  local -A wanted found
  local key
  for key in "$@"; do
    wanted[$key]=1
  done

  local line value
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%$'\r'}"
    if [[ ! "$line" =~ '^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)[[:space:]]*=[[:space:]]*(.*)$' ]]; then
      continue
    fi
    key="${match[2]}"
    (( ${+wanted[$key]} )) || continue
    value="${match[3]}"
    if [[ "$value" =~ '^"(.*)"[[:space:]]*$' || "$value" =~ "^'(.*)'[[:space:]]*\$" ]]; then
      value="${match[1]}"
    else
      # Drop an inline comment (whitespace then #), then trailing whitespace.
      value="${value%%[[:space:]]\#*}"
      if [[ "$value" =~ '^(.*[^[:space:]])?[[:space:]]+$' ]]; then
        value="${match[1]}"
      fi
    fi
    found[$key]="$value"
  done < "$env_file"

  for key in "${(k)found[@]}"; do
    if (( ! ${+parameters[$key]} )); then
      typeset -g -- "$key=${found[$key]}"
    fi
  done
}

load_diagnostic_settings() {
  load_env_file_settings "$DIAGNOSTIC_ENV_FILE" "${DIAGNOSTIC_SETTINGS[@]}"
}

# Sets `reply` to the curl arguments that add the backend API key, or to
# nothing when LLM_API_KEY is unset. The header is written to a private temp
# file and passed as `-H @file`, so the key never appears in curl's argv.
AUTH_HEADER_FILE=""
auth_header_args() {
  reply=()
  local key="${LLM_API_KEY:-}"
  [[ -n "$key" ]] || return 0

  if [[ "$key" == *[$'\r\n']* ]]; then
    print "LLM_API_KEY must not contain line breaks." >&2
    exit 1
  fi
  if [[ -z "$AUTH_HEADER_FILE" ]]; then
    make_temp_file
    AUTH_HEADER_FILE="$REPLY"
    print -r -- "Authorization: Bearer $key" > "$AUTH_HEADER_FILE"
  fi
  reply=(-H "@$AUTH_HEADER_FILE")
}

# Copies stdin to stdout with every occurrence of LLM_API_KEY masked.
redact_api_key() {
  local text
  text="$(cat)"
  if [[ -n "${LLM_API_KEY:-}" ]]; then
    text="${text//"$LLM_API_KEY"/[redacted]}"
  fi
  print -r -- "$text"
}

# Prints the first N characters of a file with the API key masked. Reading
# past N before masking keeps a key that straddles the cut from leaking.
redacted_excerpt() {
  local file="$1"
  local limit="${2:-400}"
  local key="${LLM_API_KEY:-}"
  local text
  text="$(head -c $(( limit + ${#key} )) "$file" | redact_api_key)"
  print -r -- "${text[1,limit]}"
}

# Prints a one-line hint after a 401 or 403 from an endpoint.
print_auth_hint() {
  local http_status="$1"
  [[ "$http_status" == "401" || "$http_status" == "403" ]] || return 0

  if [[ -n "${LLM_API_KEY:-}" ]]; then
    print "hint: the backend rejected the key in LLM_API_KEY; check that it is valid."
  else
    print "hint: the backend requires an API key; set LLM_API_KEY in the environment or .env."
  fi
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
