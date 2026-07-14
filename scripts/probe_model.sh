#!/bin/zsh

set -euo pipefail

SCRIPT_DIR="${0:A:h}"
source "$SCRIPT_DIR/common.sh"

if [[ $# -lt 1 ]]; then
  print "Usage: ./scripts/probe_model.sh MODEL [PROMPT] [IMAGE]" >&2
  print "  IMAGE: path to a png/jpg/jpeg/gif/webp file, or 'builtin' for the" >&2
  print "         bundled 64x64 red test square (expected reply: red)." >&2
  exit 1
fi

MODEL_ID="$1"
PROMPT_TEXT="${2:-}"
IMAGE_ARG="${3:-}"

if [[ -z "$PROMPT_TEXT" ]]; then
  if [[ -n "$IMAGE_ARG" ]]; then
    PROMPT_TEXT="Describe this image in one word."
  else
    PROMPT_TEXT="Reply with exactly ok"
  fi
fi

# 64x64 solid red PNG sent by IMAGE=builtin as a repeatable vision check.
BUILTIN_IMAGE_B64="iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9feoshgnABdLep8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3IPanc8OLDQitxAAAAAElFTkSuQmCC"

image_mime_type() {
  case "${1:l}" in
    *.png) print "image/png" ;;
    *.jpg|*.jpeg) print "image/jpeg" ;;
    *.gif) print "image/gif" ;;
    *.webp) print "image/webp" ;;
    *) return 1 ;;
  esac
}

IMAGE_B64_FILE=""
IMAGE_MIME=""
if [[ -n "$IMAGE_ARG" ]]; then
  make_temp_file
  IMAGE_B64_FILE="$REPLY"
  if [[ "$IMAGE_ARG" == "builtin" ]]; then
    IMAGE_MIME="image/png"
    print -rn -- "$BUILTIN_IMAGE_B64" > "$IMAGE_B64_FILE"
  elif [[ -f "$IMAGE_ARG" ]]; then
    if ! IMAGE_MIME="$(image_mime_type "$IMAGE_ARG")"; then
      print "Unsupported image extension (png, jpg, jpeg, gif, webp): $IMAGE_ARG" >&2
      exit 1
    fi
    base64 < "$IMAGE_ARG" | tr -d '\n' > "$IMAGE_B64_FILE"
  else
    print "Image file not found: $IMAGE_ARG" >&2
    exit 1
  fi
fi

build_payload() {
  local payload="$1"
  if [[ -n "$IMAGE_B64_FILE" ]]; then
    jq -n \
      --arg model "$MODEL_ID" \
      --arg prompt "$PROMPT_TEXT" \
      --arg mime "$IMAGE_MIME" \
      --rawfile img_b64 "$IMAGE_B64_FILE" \
      '{
        model: $model,
        messages: [{
          role: "user",
          content: [
            {type: "text", text: $prompt},
            {type: "image_url", image_url: {url: ("data:" + $mime + ";base64," + $img_b64)}}
          ]
        }]
      }' > "$payload"
  else
    jq -n \
      --arg model "$MODEL_ID" \
      --arg prompt "$PROMPT_TEXT" \
      '{model: $model, messages: [{role: "user", content: $prompt}]}' > "$payload"
  fi
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

  make_temp_file
  body="$REPLY"
  make_temp_file
  payload="$REPLY"

  build_payload "$payload"

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
if [[ -n "$IMAGE_ARG" ]]; then
  print "image: $IMAGE_ARG ($IMAGE_MIME)"
fi

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
