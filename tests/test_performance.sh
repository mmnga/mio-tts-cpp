#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BASE_URL="http://127.0.0.1:18089"
ENDPOINT="/mio/tts/stream"
REFERENCE_KEY="jp_female"
TEXT="Concurrent connection benchmark."
N_PREDICT=96
CONCURRENCY_LIST="1,2,4,8,16,32"
REQUESTS_PER_WORKER=2
CONNECT_TIMEOUT_SEC=5
MAX_TIME_SEC=180
OUTPUT_DIR="${SCRIPT_DIR}/result/perf-$(date +%Y%m%d-%H%M%S)"
SHOW_FAIL_SAMPLES=5

usage() {
  cat <<'USAGE'
Usage:
  tests/performance_test.sh [options]

Options:
  --base-url URL                 Base URL (default: http://127.0.0.1:18089)
  --endpoint PATH                Endpoint path (default: /mio/tts/stream)
  --reference-key KEY            reference_key for synthesis (default: jp_female)
  --text TEXT                    Input text (default: Concurrent connection benchmark.)
  --n-predict N                  n_predict value (default: 96)
  --concurrency LIST             Comma-separated concurrency list (default: 1,2,4,8,16,32)
  --requests-per-worker N        Requests per worker at each concurrency (default: 2)
  --connect-timeout SEC          curl connect timeout (default: 5)
  --max-time SEC                 curl max-time per request (default: 180)
  --output-dir DIR               Output directory for raw logs and summary
  --show-fail-samples N          Show up to N failed request samples per level (default: 5)
  -h, --help                     Show this help

Example:
  tests/performance_test.sh \
    --base-url http://127.0.0.1:18089 \
    --endpoint /mio/tts/stream \
    --reference-key jp_female \
    --concurrency 1,2,4,8,16 \
    --requests-per-worker 3
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-url)
      BASE_URL="${2:-}"
      shift 2
      ;;
    --endpoint)
      ENDPOINT="${2:-}"
      shift 2
      ;;
    --reference-key)
      REFERENCE_KEY="${2:-}"
      shift 2
      ;;
    --text)
      TEXT="${2:-}"
      shift 2
      ;;
    --n-predict)
      N_PREDICT="${2:-}"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY_LIST="${2:-}"
      shift 2
      ;;
    --requests-per-worker)
      REQUESTS_PER_WORKER="${2:-}"
      shift 2
      ;;
    --connect-timeout)
      CONNECT_TIMEOUT_SEC="${2:-}"
      shift 2
      ;;
    --max-time)
      MAX_TIME_SEC="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --show-fail-samples)
      SHOW_FAIL_SAMPLES="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "required command not found: $cmd" >&2
    exit 1
  fi
}

require_cmd curl
require_cmd awk
require_cmd sort
require_cmd xargs
require_cmd seq

if [[ ! "$N_PREDICT" =~ ^[0-9]+$ ]]; then
  echo "--n-predict must be an integer: $N_PREDICT" >&2
  exit 1
fi
if [[ ! "$REQUESTS_PER_WORKER" =~ ^[0-9]+$ ]] || [[ "$REQUESTS_PER_WORKER" -lt 1 ]]; then
  echo "--requests-per-worker must be >= 1: $REQUESTS_PER_WORKER" >&2
  exit 1
fi
if [[ ! "$CONNECT_TIMEOUT_SEC" =~ ^[0-9]+$ ]] || [[ "$CONNECT_TIMEOUT_SEC" -lt 1 ]]; then
  echo "--connect-timeout must be >= 1: $CONNECT_TIMEOUT_SEC" >&2
  exit 1
fi
if [[ ! "$MAX_TIME_SEC" =~ ^[0-9]+$ ]] || [[ "$MAX_TIME_SEC" -lt 1 ]]; then
  echo "--max-time must be >= 1: $MAX_TIME_SEC" >&2
  exit 1
fi
if [[ ! "$SHOW_FAIL_SAMPLES" =~ ^[0-9]+$ ]]; then
  echo "--show-fail-samples must be an integer: $SHOW_FAIL_SAMPLES" >&2
  exit 1
fi

if [[ "${ENDPOINT:0:1}" != "/" ]]; then
  ENDPOINT="/${ENDPOINT}"
fi
TARGET_URL="${BASE_URL%/}${ENDPOINT}"

mkdir -p "$OUTPUT_DIR"

json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  s="${s//$'\n'/\\n}"
  s="${s//$'\r'/\\r}"
  s="${s//$'\t'/\\t}"
  printf '%s' "$s"
}

PAYLOAD="$(printf '{"text":"%s","reference_key":"%s","n_predict":%d}' \
  "$(json_escape "$TEXT")" \
  "$(json_escape "$REFERENCE_KEY")" \
  "$N_PREDICT")"

echo "== mio-tts concurrent benchmark =="
echo "target:        $TARGET_URL"
echo "reference_key: $REFERENCE_KEY"
echo "n_predict:     $N_PREDICT"
echo "output_dir:    $OUTPUT_DIR"

if ! curl -fsS --connect-timeout "$CONNECT_TIMEOUT_SEC" --max-time "$MAX_TIME_SEC" \
  "${BASE_URL%/}/mio/references" >/dev/null; then
  echo "preflight failed: cannot reach ${BASE_URL%/}/mio/references" >&2
  exit 1
fi

REF_JSON="$(curl -sS --connect-timeout "$CONNECT_TIMEOUT_SEC" --max-time "$MAX_TIME_SEC" \
  "${BASE_URL%/}/mio/references" || true)"
if [[ "$REF_JSON" != *"\"key\":\"$REFERENCE_KEY\""* ]]; then
  echo "warning: reference_key '$REFERENCE_KEY' was not found in /mio/references response" >&2
fi

run_one_request() {
  local req_id="$1"
  local result
  local rc=0
  result="$(
    curl -sS -o /dev/null \
      --connect-timeout "$CONNECT_TIMEOUT_SEC" \
      --max-time "$MAX_TIME_SEC" \
      -H 'Content-Type: application/json' \
      -d "$PAYLOAD" \
      -w '%{http_code},%{time_total},%{time_connect},%{time_starttransfer},%{size_download}' \
      "$TARGET_URL" 2>/dev/null
  )" || rc=$?

  if [[ -z "$result" ]]; then
    result="000,0,0,0,0"
  fi

  printf '%s,%s,%s\n' "$req_id" "$result" "$rc"
}

percentile_ms() {
  local sorted_file="$1"
  local p="$2"
  awk -v p="$p" '
    { a[++n] = $1 }
    END {
      if (n == 0) {
        print "NA"
        exit
      }
      idx = int((p/100.0) * n + 0.999999)
      if (idx < 1) idx = 1
      if (idx > n) idx = n
      printf "%.2f", a[idx] * 1000.0
    }
  ' "$sorted_file"
}

export TARGET_URL PAYLOAD CONNECT_TIMEOUT_SEC MAX_TIME_SEC
export -f run_one_request

SUMMARY_CSV="${OUTPUT_DIR}/summary.csv"
{
  echo "concurrency,total,ok,failed,success_rate_pct,wall_sec,rps_ok,lat_avg_ms,lat_p50_ms,lat_p90_ms,lat_p95_ms,lat_p99_ms,ttfb_avg_ms"
} > "$SUMMARY_CSV"

IFS=',' read -r -a CONCURRENCIES <<< "$CONCURRENCY_LIST"

printf '\n%-11s %-8s %-8s %-8s %-10s %-10s %-10s %-10s %-9s %-9s %-9s %-9s %-10s\n' \
  "concurrency" "total" "ok" "failed" "success%" "wall(s)" "rps(ok)" "avg(ms)" "p50(ms)" "p90(ms)" "p95(ms)" "p99(ms)" "ttfb(ms)"

for raw_c in "${CONCURRENCIES[@]}"; do
  c="$(echo "$raw_c" | tr -d '[:space:]')"
  if [[ -z "$c" ]]; then
    continue
  fi
  if [[ ! "$c" =~ ^[0-9]+$ ]] || [[ "$c" -lt 1 ]]; then
    echo "invalid concurrency value: $raw_c" >&2
    exit 1
  fi

  total_requests=$((c * REQUESTS_PER_WORKER))
  raw_file="${OUTPUT_DIR}/raw_c${c}.csv"
  lat_ok_file="${OUTPUT_DIR}/lat_ok_c${c}.txt"

  echo "running: concurrency=${c}, total_requests=${total_requests}"
  t0_ns="$(date +%s%N)"
  seq 1 "$total_requests" | xargs -n 1 -P "$c" bash -c 'run_one_request "$1"' _ > "$raw_file"
  t1_ns="$(date +%s%N)"

  ok_count="$(awk -F',' '$2 == 200 && $7 == 0 {ok++} END {print ok+0}' "$raw_file")"
  failed_count=$((total_requests - ok_count))

  wall_sec="$(awk -v t0="$t0_ns" -v t1="$t1_ns" 'BEGIN { printf "%.4f", (t1 - t0) / 1000000000.0 }')"
  success_rate="$(awk -v ok="$ok_count" -v total="$total_requests" 'BEGIN { if (total == 0) { print "0.00" } else { printf "%.2f", (ok / total) * 100.0 } }')"
  rps_ok="$(awk -v ok="$ok_count" -v wall="$wall_sec" 'BEGIN { if (wall <= 0) { print "0.00" } else { printf "%.2f", ok / wall } }')"

  awk -F',' '$2 == 200 && $7 == 0 { printf "%.6f\n", $3 }' "$raw_file" | sort -n > "$lat_ok_file"

  lat_avg_ms="$(awk -F',' '$2 == 200 && $7 == 0 {sum += $3; n++} END {if (n == 0) print "NA"; else printf "%.2f", (sum / n) * 1000.0}' "$raw_file")"
  ttfb_avg_ms="$(awk -F',' '$2 == 200 && $7 == 0 {sum += $5; n++} END {if (n == 0) print "NA"; else printf "%.2f", (sum / n) * 1000.0}' "$raw_file")"
  p50_ms="$(percentile_ms "$lat_ok_file" 50)"
  p90_ms="$(percentile_ms "$lat_ok_file" 90)"
  p95_ms="$(percentile_ms "$lat_ok_file" 95)"
  p99_ms="$(percentile_ms "$lat_ok_file" 99)"

  printf '%-11s %-8s %-8s %-8s %-10s %-10s %-10s %-10s %-9s %-9s %-9s %-9s %-10s\n' \
    "$c" "$total_requests" "$ok_count" "$failed_count" "${success_rate}%" "$wall_sec" "$rps_ok" "$lat_avg_ms" "$p50_ms" "$p90_ms" "$p95_ms" "$p99_ms" "$ttfb_avg_ms"

  {
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$c" "$total_requests" "$ok_count" "$failed_count" "$success_rate" "$wall_sec" "$rps_ok" \
      "$lat_avg_ms" "$p50_ms" "$p90_ms" "$p95_ms" "$p99_ms" "$ttfb_avg_ms"
  } >> "$SUMMARY_CSV"

  if [[ "$failed_count" -gt 0 ]] && [[ "$SHOW_FAIL_SAMPLES" -gt 0 ]]; then
    echo "  failed samples (up to ${SHOW_FAIL_SAMPLES}):"
    awk -F',' '$2 != 200 || $7 != 0 { print "    req_id=" $1 ", http=" $2 ", total_s=" $3 ", rc=" $7 }' "$raw_file" | head -n "$SHOW_FAIL_SAMPLES"
  fi
done

echo
echo "summary csv: $SUMMARY_CSV"
echo "raw per-level files: ${OUTPUT_DIR}/raw_c*.csv"
