#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---------- paths ----------
BIN="${REPO_ROOT}/build/llama-tts-mio"
MODEL_DIR="${REPO_ROOT}/models"
LLM_MODEL="${MODEL_DIR}/MioTTS-0.1B-Q8_0.gguf"
VOCODER_MODEL="${MODEL_DIR}/miocodec.gguf"
WAVLM_MODEL="${MODEL_DIR}/wavlm_base_plus_2l_f32.gguf"
DEFAULT_EMB="${MODEL_DIR}/jp_female.emb.gguf"
REFERENCE_AUDIO="resources/ja_female.wav"
TMP_DIR="${SCRIPT_DIR}/result/cli-test-$(date +%Y%m%d-%H%M%S)"

# ---------- colors ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# ---------- counters ----------
PASSED=0
FAILED=0
SKIPPED=0

mkdir -p "$TMP_DIR"

# ---------- helpers ----------
log_info()  { printf "${CYAN}[INFO]${NC}  %s\n" "$*"; }
log_pass()  { printf "${GREEN}[PASS]${NC}  %s\n" "$*"; ((PASSED++)); }
log_fail()  { printf "${RED}[FAIL]${NC}  %s\n" "$*"; ((FAILED++)); }
log_skip()  { printf "${YELLOW}[SKIP]${NC}  %s\n" "$*"; ((SKIPPED++)); }

check_file_exists() {
    local path="$1"
    local label="$2"
    if [[ ! -f "$path" ]]; then
        log_fail "$label: output file not found: $path"
        return 1
    fi
    local size
    size=$(stat -f%z "$path" 2>/dev/null || stat -c%s "$path" 2>/dev/null || echo 0)
    if [[ "$size" -lt 100 ]]; then
        log_fail "$label: output file too small (${size} bytes): $path"
        return 1
    fi
    return 0
}

check_wav_valid() {
    local path="$1"
    local label="$2"
    if ! file "$path" | grep -q "WAVE audio"; then
        log_fail "$label: not a valid WAV file: $path"
        return 1
    fi
    return 0
}

run_timed() {
    local label="$1"
    shift
    log_info "$label: running..."
    local t0 t1 elapsed rc=0
    t0=$(date +%s%N)
    "$@" > "${TMP_DIR}/stdout.log" 2> "${TMP_DIR}/stderr.log" || rc=$?
    t1=$(date +%s%N)
    elapsed=$(awk -v t0="$t0" -v t1="$t1" 'BEGIN { printf "%.1f", (t1 - t0) / 1000000000.0 }')
    if [[ $rc -ne 0 ]]; then
        log_fail "$label: exit code $rc (${elapsed}s)"
        if [[ -s "${TMP_DIR}/stderr.log" ]]; then
            printf "  stderr (last 10 lines):\n"
            tail -10 "${TMP_DIR}/stderr.log" | sed 's/^/    /'
        fi
        return 1
    fi
    log_info "$label: completed in ${elapsed}s"
    return 0
}

# ---------- preflight ----------
log_info "=== llama-tts-mio CLI test suite ==="
log_info "tmp dir: $TMP_DIR"

if [[ ! -x "$BIN" ]]; then
    echo "ERROR: binary not found: $BIN" >&2
    echo "Run build.sh first." >&2
    exit 1
fi

for f in "$LLM_MODEL" "$VOCODER_MODEL"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: required model not found: $f" >&2
        exit 1
    fi
done

# ==========================================================
# Test 1: Basic synthesis (Japanese, with pre-built embedding)
# ==========================================================
TEST1_OUT="${TMP_DIR}/test1_basic.wav"
if [[ -f "$DEFAULT_EMB" ]]; then
    if run_timed "Test 1: Basic synthesis (JP)" \
        "$BIN" \
            -m "$LLM_MODEL" \
            -mv "$VOCODER_MODEL" \
            -emb "$DEFAULT_EMB" \
            -p "こんにちは、テストです。" \
            -o "$TEST1_OUT" \
            -n 200 \
            --threads 4 \
            --flash-attn on; then
        if check_file_exists "$TEST1_OUT" "Test 1" && check_wav_valid "$TEST1_OUT" "Test 1"; then
            log_pass "Test 1: Basic synthesis (JP) -> $TEST1_OUT"
        fi
    fi
else
    log_skip "Test 1: default embedding not found: $DEFAULT_EMB"
fi

# ==========================================================
# Test 2: Basic synthesis (English)
# ==========================================================
TEST2_OUT="${TMP_DIR}/test2_english.wav"
EN_EMB="${MODEL_DIR}/en_female.emb.gguf"
if [[ -f "$EN_EMB" ]]; then
    if run_timed "Test 2: Basic synthesis (EN)" \
        "$BIN" \
            -m "$LLM_MODEL" \
            -mv "$VOCODER_MODEL" \
            -emb "$EN_EMB" \
            -p "Hello, this is a test." \
            -o "$TEST2_OUT" \
            -n 200 \
            --threads 4 \
            --flash-attn on; then
        if check_file_exists "$TEST2_OUT" "Test 2" && check_wav_valid "$TEST2_OUT" "Test 2"; then
            log_pass "Test 2: Basic synthesis (EN) -> $TEST2_OUT"
        fi
    fi
else
    log_skip "Test 2: english embedding not found: $EN_EMB"
fi

# ==========================================================
# Test 3: Create reference from audio (embedding generation)
# ==========================================================
TEST3_EMB_OUT="${TMP_DIR}/test3_reference.emb.gguf"
if [[ -f "$REFERENCE_AUDIO" ]] && [[ -f "$WAVLM_MODEL" ]]; then
    if run_timed "Test 3: Create reference embedding" \
        "$BIN" \
            -mv "$VOCODER_MODEL" \
            --tts-wavlm-model "$WAVLM_MODEL" \
            --tts-reference-audio "$REFERENCE_AUDIO" \
            --tts-mio-embedding-out "$TEST3_EMB_OUT" \
            --tts-mio-embedding-only \
            --threads 4 \
            --flash-attn on; then
        if check_file_exists "$TEST3_EMB_OUT" "Test 3"; then
            log_pass "Test 3: Create reference embedding -> $TEST3_EMB_OUT"
        fi
    fi
else
    [[ ! -f "$REFERENCE_AUDIO" ]] && log_skip "Test 3: reference audio not found: $REFERENCE_AUDIO"
    [[ ! -f "$WAVLM_MODEL" ]]     && log_skip "Test 3: wavlm model not found: $WAVLM_MODEL"
fi

# ==========================================================
# Test 4: Synthesis with newly created reference
# ==========================================================
TEST4_OUT="${TMP_DIR}/test4_custom_ref.wav"
if [[ -f "$TEST3_EMB_OUT" ]]; then
    if run_timed "Test 4: Synthesis with custom reference" \
        "$BIN" \
            -m "$LLM_MODEL" \
            -mv "$VOCODER_MODEL" \
            -emb "$TEST3_EMB_OUT" \
            -p "カスタムリファレンスでの音声合成テストです。" \
            -o "$TEST4_OUT" \
            -n 200 \
            --threads 4 \
            --flash-attn on; then
        if check_file_exists "$TEST4_OUT" "Test 4" && check_wav_valid "$TEST4_OUT" "Test 4"; then
            log_pass "Test 4: Synthesis with custom reference -> $TEST4_OUT"
        fi
    fi
else
    log_skip "Test 4: requires Test 3 embedding (skipped or failed)"
fi

# ==========================================================
# Test 5: Codes-only mode (generate codes without audio)
# ==========================================================
TEST5_CODES="${TMP_DIR}/test5_codes.txt"
if [[ -f "$DEFAULT_EMB" ]]; then
    if run_timed "Test 5: Codes-only mode" \
        "$BIN" \
            -m "$LLM_MODEL" \
            -mv "$VOCODER_MODEL" \
            -emb "$DEFAULT_EMB" \
            -p "コード生成テスト。" \
            --tts-mio-codes-out "$TEST5_CODES" \
            --tts-mio-codes-only \
            -n 100 \
            --threads 4 \
            --flash-attn on; then
        if check_file_exists "$TEST5_CODES" "Test 5"; then
            local_lines=$(wc -l < "$TEST5_CODES" | tr -d ' ')
            log_info "Test 5: generated codes file has ${local_lines} line(s)"
            log_pass "Test 5: Codes-only mode -> $TEST5_CODES"
        fi
    fi
else
    log_skip "Test 5: default embedding not found: $DEFAULT_EMB"
fi

# ==========================================================
# Test 6: Decode from codes file
# ==========================================================
TEST6_OUT="${TMP_DIR}/test6_from_codes.wav"
if [[ -f "$TEST5_CODES" ]]; then
    if run_timed "Test 6: Decode from codes file" \
        "$BIN" \
            -mv "$VOCODER_MODEL" \
            -emb "$DEFAULT_EMB" \
            --tts-mio-codes-in "$TEST5_CODES" \
            -o "$TEST6_OUT" \
            --threads 4 \
            --flash-attn on; then
        if check_file_exists "$TEST6_OUT" "Test 6" && check_wav_valid "$TEST6_OUT" "Test 6"; then
            log_pass "Test 6: Decode from codes file -> $TEST6_OUT"
        fi
    fi
else
    log_skip "Test 6: requires Test 5 codes file (skipped or failed)"
fi

# ==========================================================
# Test 7: Long text synthesis
# ==========================================================
TEST7_OUT="${TMP_DIR}/test7_long.wav"
LONG_TEXT="これは長文テストです。音声合成エンジンが長い文章を正しく処理できるか確認します。複数の文を含むテキストを生成して、安定性を検証します。"
if [[ -f "$DEFAULT_EMB" ]]; then
    if run_timed "Test 7: Long text synthesis" \
        "$BIN" \
            -m "$LLM_MODEL" \
            -mv "$VOCODER_MODEL" \
            -emb "$DEFAULT_EMB" \
            -p "$LONG_TEXT" \
            -o "$TEST7_OUT" \
            -n 400 \
            --threads 4 \
            --flash-attn on; then
        if check_file_exists "$TEST7_OUT" "Test 7" && check_wav_valid "$TEST7_OUT" "Test 7"; then
            log_pass "Test 7: Long text synthesis -> $TEST7_OUT"
        fi
    fi
else
    log_skip "Test 7: default embedding not found: $DEFAULT_EMB"
fi

# ==========================================================
# Test 8: Reference from audio + synthesis in one shot
# ==========================================================
TEST8_OUT="${TMP_DIR}/test8_ref_and_synth.wav"
if [[ -f "$REFERENCE_AUDIO" ]] && [[ -f "$WAVLM_MODEL" ]]; then
    if run_timed "Test 8: Reference + synthesis (one shot)" \
        "$BIN" \
            -m "$LLM_MODEL" \
            -mv "$VOCODER_MODEL" \
            --tts-wavlm-model "$WAVLM_MODEL" \
            --tts-reference-audio "$REFERENCE_AUDIO" \
            -p "リファレンスから一発で合成するテストです。" \
            -o "$TEST8_OUT" \
            -n 200 \
            --threads 4 \
            --flash-attn on; then
        if check_file_exists "$TEST8_OUT" "Test 8" && check_wav_valid "$TEST8_OUT" "Test 8"; then
            log_pass "Test 8: Reference + synthesis (one shot) -> $TEST8_OUT"
        fi
    fi
else
    [[ ! -f "$REFERENCE_AUDIO" ]] && log_skip "Test 8: reference audio not found"
    [[ ! -f "$WAVLM_MODEL" ]]     && log_skip "Test 8: wavlm model not found"
fi

# ==========================================================
# Summary
# ==========================================================
echo
log_info "========================================="
log_info "  Results: ${GREEN}${PASSED} passed${NC}, ${RED}${FAILED} failed${NC}, ${YELLOW}${SKIPPED} skipped${NC}"
log_info "  Output:  ${TMP_DIR}"
log_info "========================================="

if [[ "$FAILED" -gt 0 ]]; then
    exit 1
fi
exit 0
