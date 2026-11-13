#!/usr/bin/env bash
# Serve a Tandem lite package (DeepSeek-V4.1-Flash, VQ experts) on one GPU, or on 2/4/8 (GPU=0,1).
# The package's config.json "tandem" block sets the expert stores (planes,
# VQ table, FP4 hot set; vllm tandem_pkg.py); this script sets the runtime:
# host-resident VQ base behind a GPU cache, misses read in-GEMM from pinned
# host rows (DIRECT) and computed by the CPU (Tandem), exact (no skipped experts).
#
#   MODEL=<package dir>   (required; the whole HF repo, downloaded)
#   FP4_TOP=K             static FP4 experts per layer from the hot-set store
#                         (0 = VQ only; any K <= 64; the FP4 pool size follows)
#   BASE_GB=N             GPU cache for the VQ base (GiB; the rest of the card
#                         after KV, FP4 pool and activations)
#   GPU=0,1               several GPUs (2, 4 or 8): vLLM tensor parallel for attention, Engram and the shared
#                         experts; each GPU owns whole routed experts (VLLM_MOE_W2_EP). BASE_GB and KV_GB are
#                         per GPU, RAM_GB is the total (split over the GPUs). ET1: 2x PRO 6000 +18% vs one GPU;
#                         an 8-core CPU is too small for two (rome RB5: slower than one GPU)
#   HELPER_GPU=1 HELPER_GB=40
#                         one main GPU (GPU=) + a helper GPU that holds a fixed warm slice of the ranked experts
#                         (HELPER_GB GiB after the main cache's share) and computes the main GPU's misses on them
#                         next to the CPU; the CPU keeps the rest. For hosts whose CPU is too small for TP
#                         (VLLM_MOE_W2_CPU_HELPER, VLLM_MOE_W2_HELPER_GB). HELPER_SPLIT=alt: the helper owns every
#                         other expert of the top ranks instead and computes all routes to them; the main GPU
#                         caches only the rest (the two GPUs split the hot set evenly). HELPER_SPLIT=dyn: that
#                         layout as the start, then the helper caches the hottest experts the main GPU does not
#                         serve (loaded from RAM over its own link; exclusive with the main GPU). dyn is the default
#                         (rome RH5/RH6: +9%/+11% vs alt; 24 GB cards: HELPER_GB=22 fits, +30% vs one card); set
#                         HELPER_SPLIT=slice for the fixed slice. No GPU P2P needed (data goes through pinned RAM)
#   GPU=0 PORT=8000 KV_GB=4 MAX_LEN=32768 NUM_SEQS=STEP_T/(1+k) CPU=1
#                         (MAX_LEN is the per-request limit; the KV_GB=4 cache holds about 140k tokens shared by all
#                         streams, so it costs no VRAM. 32768 tested on the 5090 and the 96 GB card, 131072 with KV_GB=8)
#   THINKING=1 REASONING_EFFORT=low|high|max   chat-template defaults
#   TOOLS=1               OpenAI tool calling (--enable-auto-tool-choice, deepseek_v41 tool parser) and the
#                         thinking text in its own reasoning field (deepseek_v41 reasoning parser); 0 = both off
#                         (PT1: tool calls, tool results, streaming, thinking on/off checked)
#   IMAGES=N              image inputs per prompt (vision encoder on): +1.0 GiB of VRAM at boot, +1.5 GiB at
#                         peak with a max-size image (VIS1/VIS2), so lower BASE_GB by 2; default 0 = encoder not loaded
#   CUBIT_DIR=<dir>       SM120 kernel cubins (VLLM_MOE_W2_CUBIT_DIR)
#   RAM_GB=N              small-RAM box: pin only the N GiB of most-routed VQ experts,
#                         read the rest from the package files (VLLM_TANDEM_RAM_GB)
#   ENGRAM=files          read Engram rows from the package files on demand instead of
#                         pinning the tables (97 GiB) in RAM (VLLM_TANDEM_ENGRAM_FILES)
#   SPEC_N=k              DSpark speculative decoding with the drafter in the package (mtp layers), k tokens;
#                         verify steps must fit the decode graphs ((1+k)*NUM_SEQS <= STEP_T), so NUM_SEQS
#                         defaults to 8/(1+k) (k=3: 2)
#   STEP_T=N              largest decode step in tokens (seqs x (1+k)) the CUDA graphs cover; default 32 (up to 32
#                         streams) without a drafter on cards with 48 GB or more, else 8 (smaller cards, or SPEC_N:
#                         a larger step takes GPU memory from the expert cache)
#   CHUNK=N               tokens per scheduler step (prefill chunk; --max-num-batched-tokens), default 4096
#                         (PF1/PF2: 1.6x prefill vs 2048, survives 94k prompts at BASE 65.5 on 96 GB; 8192 needs
#                         BASE <= 60, 16384 BASE <= 54. KV is fixed, so vLLM does not profile the prefill peak: a
#                         too-large CHUNK + BASE pair boots, then OOMs on the first long prompt)
#   VENV=<venv>           python env with the Tandem vLLM build
#   HOST=<addr>           bind address (default: all interfaces; 127.0.0.1 = this machine only)
#   ALLOW=<cidr,...>      serve only these clients, e.g. ALLOW=10.0.0.0/24,192.168.1.7 (loopback always
#                         allowed; 403 otherwise). Not a firewall: the connection is accepted first
#   PANEL=1               status page at http://<host>:<port>/tandem/ (counters only; with an API key
#                         open /tandem/#key=<key>); combine with ALLOW= beyond this machine
# Extra arguments go to `vllm serve`.
set -euo pipefail
MODEL=${MODEL:?set MODEL=<Tandem package dir>}
[ -f "$MODEL/config.json" ] && grep -q '"tandem"' "$MODEL/config.json" \
  || { echo "$MODEL: no Tandem package (config.json has no \"tandem\" block)" >&2; exit 2; }
VENV=${VENV:-$HOME/.venvs/tandem}
export PATH="$VENV/bin:$PATH"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=${GPU:-0}
if [ -n "${HELPER_GPU:-}" ]; then
  case "${GPU:-0}" in *,*) echo "HELPER_GPU needs one main GPU (GPU=${GPU})" >&2; exit 2 ;; esac
  export CUDA_VISIBLE_DEVICES=${GPU:-0},$HELPER_GPU VLLM_MOE_W2_CPU_HELPER=1
  export VLLM_MOE_W2_HELPER_GB=${HELPER_GB:-20} VLLM_MOE_W2_HELPER_SPLIT=${HELPER_SPLIT:-dyn}
fi

# expert stores: from the package (engine side); FP4 count
export VLLM_TANDEM_FP4_TOP=${FP4_TOP:-0}
[ "$VLLM_TANDEM_FP4_TOP" -gt 0 ] && export VLLM_MOE_W2_PREFILL_FP4=1   # prefill reads the FP4 tier too
export VLLM_MOE_W2_CUBIT_DIR=${CUBIT_DIR:-${VLLM_MOE_W2_CUBIT_DIR:-$(cd "$(dirname "$0")" && pwd)/cubins}}
export VLLM_MOE_W2_KS=7168,6144,5120,4096,2304,2048,1152,1024,512
# several GPUs: tensor parallel, whole experts per GPU; RAM_GB is shared out per GPU process
NGPU=$(echo "${GPU:-0}" | tr ',' '\n' | grep -c .)
PAR=()
if [ "$NGPU" -gt 1 ]; then
  case $NGPU in 2|4|8) ;; *) echo "GPU=${GPU}: use 1, 2, 4 or 8 GPUs" >&2; exit 2 ;; esac
  export VLLM_MOE_W2_EP=1
  PAR=(--tensor-parallel-size "$NGPU")
fi
[ -n "${RAM_GB:-}" ] && export VLLM_TANDEM_RAM_GB=$(awk "BEGIN{print ${RAM_GB} / ${NGPU}}")
EG='{"use_thp":true}'
[ "${ENGRAM:-ram}" = files ] && export VLLM_TANDEM_ENGRAM_FILES=1 && EG='{"cpu_offload":true}'

# VQ base: host-resident, GPU cache (decayed-frequency pool), DIRECT misses + CPU experts
export VLLM_MOE_W2_BASE_CACHE_GB=${BASE_GB:?set BASE_GB=<GiB GPU cache for the VQ base>}
export VLLM_MOE_W2_BASE_DIRECT=1 VLLM_MOE_W2_CPU=${CPU:-1}
export VLLM_MOE_W2_BASE_POLICY=freq VLLM_MOE_W2_PROMO_HYST=1.25 VLLM_MOE_W2_DELTA_DECAY_S=30
export VLLM_MOE_W2_FORCE_POOL=1 VLLM_MOE_W2_DELTA_PIN=1 VLLM_MOE_W2_DELTA_POLICY=need
export VLLM_MOE_W2_REPLAY_MODE=strict VLLM_MOE_W2_BASE_MISS_TOL=0
# BT1-BT4: 32 costs nothing without a drafter on a 96 GB card; on a 32 GB card its workspace does not fit next to
# the expert cache (pcbox1 5090: warm-up OOM at BASE 10), so cards under 48 GB default to 8 (up to 8 streams)
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "${GPU:-0}" 2>/dev/null | head -1 || true)
STEP_T=${STEP_T:-$([ -n "${SPEC_N:-}" ] || [ "${GPU_MIB:-99999}" -lt 49152 ] && echo 8 || echo 32)}
# SM89 (Ada): vLLM ranks Marlin weight-only FP8 first there; it repacks wo_a and breaks the o_proj bmm (RB2)
GPU_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader -i "${GPU:-0}" 2>/dev/null | head -1 || true)
[ "$GPU_CC" = 8.9 ] && export VLLM_DISABLED_KERNELS=${VLLM_DISABLED_KERNELS:-MarlinFP8ScaledMMLinearKernel,HummingFP8ScaledMMLinearKernel}
export VLLM_MOE_W2_PREFILL_T=$STEP_T VLLM_MOE_W2_GATE=0
export VLLM_MOE_W2_STABLE_TOPK=1 VLLM_MOE_W2_ZERO_NULL_BLOCK=1
export VLLM_USE_V2_MODEL_RUNNER=1 VLLM_USE_BREAKABLE_CUDAGRAPH=0
export VLLM_ENGINE_READY_TIMEOUT_S=2400

# status panel (vllm/entrypoints/serve/middleware/tandem_panel.py); the engine
# writes its counters to VLLM_TANDEM_STATS (quantization/utils/tandem_stats.py)
MW=()
if [ "${PANEL:-0}" = 1 ]; then
  export VLLM_TANDEM_STATS=/dev/shm/tandem-stats-${PORT:-8000}.json
  export VLLM_TANDEM_MODEL=$MODEL
  export VLLM_TANDEM_SHAPE="RAM ${RAM_GB:-all} GiB pinned, GPU cache ${BASE_GB} GiB"
  MW+=(--middleware vllm.entrypoints.serve.middleware.tandem_panel.TandemPanelMiddleware)
fi
# client IP allowlist (vllm/entrypoints/serve/middleware/ip_allow.py); added
# last = outermost, so it also guards the panel
if [ -n "${ALLOW:-}" ]; then
  export VLLM_TANDEM_ALLOW=$ALLOW
  MW+=(--middleware vllm.entrypoints.serve.middleware.ip_allow.IPAllowMiddleware)
fi

# decode graphs cover up to STEP_T tokens per step (PREFILL_T); a DSpark verify step is (1+k) tokens per sequence
NUM_SEQS=${NUM_SEQS:-$(( STEP_T / (1 + ${SPEC_N:-0}) ))}
CAP=$(for n in 1 2 3 4 5 6 8 12 16 24 32 48 64; do if [ "$n" -le "$STEP_T" ]; then echo -n "$n,"; fi; done)
CAP="[${CAP%,}]"

if [ "${THINKING:-1}" = 1 ]; then
  CT="{\"thinking\":true,\"reasoning_effort\":\"${REASONING_EFFORT:-low}\"}"
else
  CT='{"thinking":false}'
fi
TOOL_ARGS=()
[ "${TOOLS:-1}" = 1 ] && TOOL_ARGS=(--enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41)
exec vllm serve "$MODEL" --served-model-name "${SERVED_NAME:-deepseek-v4-flash}" --trust-remote-code \
  --tokenizer-mode deepseek_v41 --limit-mm-per-prompt "{\"image\":${IMAGES:-0}}" --engram-config "$EG" \
  --kv-cache-dtype fp8 --max-model-len "${MAX_LEN:-32768}" \
  --kv-cache-memory-bytes $(( ${KV_GB:-4} * 1073741824 )) --gpu-memory-utilization "${GPU_UTIL:-0.97}" \
  --max-num-batched-tokens "${CHUNK:-4096}" --max-num-seqs "$NUM_SEQS" --no-scheduler-reserve-full-isl \
  --default-chat-template-kwargs "$CT" "${TOOL_ARGS[@]}" "${PAR[@]}" \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"],"cudagraph_capture_sizes":'"$CAP"'}' \
  ${SPEC_N:+--speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":${SPEC_N}}"} \
  ${HOST:+--host "$HOST"} "${MW[@]}" --port "${PORT:-8000}" "$@"
