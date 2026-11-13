#!/usr/bin/env bash
# Install Tandem-vLLM into a fresh Python venv (no Docker, no root).
#
#   tools/install_tandem.sh [--prefix DIR] [--src DIR] [--python python3.12] [--check] [--no-link]
#
#   --prefix  venv to create (default ~/.venvs/tandem); must not hold another vLLM
#   --src     engine source: the release tree's engine/ (default when present), a tree with
#             TANDEM_FILES, or a vllm-v0.31.0 checkout of branch moet-v0.31.0 (dev layout)
#   --check   only check the prerequisites
#
# Steps:
#   1. prerequisites: Linux x86-64, NVIDIA driver + a supported GPU: SM120 (RTX 50xx / RTX PRO
#      6000 Blackwell), SM90 (H100/H200), SM89 (RTX 40xx / RTX 6000 Ada / L40), SM86 (RTX 30xx /
#      RTX A6000). CUDA toolkit with nvcc
#      (FlashInfer builds its kernels at first use), Python 3.12 with venv, or uv (it can fetch
#      Python 3.12 itself), g++ (the CPU expert engine is compiled at first use, -march=native),
#      a CPU with AVX2 (AVX-512 is used when present)
#   2. venv + the stock vLLM 0.31.0 wheel from PyPI (pulls torch, FlashInfer 0.7.0.post1);
#      made with python -m venv + pip, or with uv when Python 3.12's venv module is missing
#   3. Tandem engine files copied over the wheel's (originals kept as <file>.orig)
#   4. DeepSeek-V4.1 SM120 kernel fixes: FlashInfer page-32 sparse-MLA sources
#      (patch_flashinfer_dsv41.py) and DeepGEMM _C with 128-row indexer pages (prebuilt)
#   5. SM120 cubins -> <prefix>/share/tandem/cubins
#   6. <prefix>/bin/tandem-serve: the package launcher with this venv, the cubins and the
#      CUDA toolkit on PATH (non-interactive shells often lack /usr/local/cuda/bin), plus a
#      link ~/.local/bin/tandem-serve (not replaced if it points elsewhere; --no-link skips it)
#
# Then: tandem-serve MODEL=<package dir> BASE_GB=<GiB> ... (see tools/tandem_serve.sh).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PREFIX=$HOME/.venvs/tandem
# release layout (engine/, deps/deepgemm/, cubins/ next to tools/), else the dev layout
if [ -d "$HERE/../engine" ]; then SRC=$HERE/../engine; else SRC=$HERE/../vllm-v0.31.0; fi
PY=python3.12
CHECK=0
while [ $# -gt 0 ]; do
  case $1 in
    --prefix) PREFIX=$2; shift 2 ;;
    --no-link) NOLINK=1; shift ;;
    --src) SRC=$2; shift 2 ;;
    --python) PY=$2; shift 2 ;;
    --check) CHECK=1; shift ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
VLLM_VER=0.31.0
FI_VER=0.7.0.post1
DGN=_C.cpython-312-x86_64-linux-gnu.so
if [ -f "$HERE/../deps/deepgemm/$DGN" ]; then DG_DEF=$HERE/../deps/deepgemm/$DGN
else DG_DEF=$HERE/../deps/deepgemm-C-out/$DGN; fi
if [ -d "$HERE/../cubins" ]; then CUB_DEF=$HERE/../cubins; else CUB_DEF=$HOME/moe-cache/cubins-sm120; fi
DG_SO=${DG_SO:-$DG_DEF}
CUBINS=${CUBINS:-$CUB_DEF}

say() { echo "[tandem-install] $*"; }
die() { echo "[tandem-install] ERROR: $*" >&2; exit 1; }

# ---- 1. prerequisites
[ "$(uname -s)-$(uname -m)" = Linux-x86_64 ] || die "Linux x86-64 only"
command -v nvidia-smi > /dev/null || die "nvidia-smi not found (NVIDIA driver)"
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | tr '\n' ' ')
GOOD=0
for c in $CC; do
  case $c in
    12.0|9.0|8.9|8.6) GOOD=1 ;;
    *) say "warning: GPU with compute capability $c is not supported; do not use it with Tandem" ;;
  esac
done
[ "$GOOD" = 1 ] || die "no supported GPU (compute capability: $CC); Tandem runs on SM120 (RTX 50xx / PRO 6000 Blackwell), SM90 (H100/H200), SM89 (RTX 40xx / Ada), SM86 (RTX 30xx / A6000)"
NVCC=$(command -v nvcc || true)
for d in ${CUDA_HOME:-} /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-13; do
  [ -z "$NVCC" ] && [ -n "$d" ] && [ -x "$d/bin/nvcc" ] && NVCC=$d/bin/nvcc
done
[ -n "$NVCC" ] || die "nvcc not found: install the CUDA toolkit (13.x) or set CUDA_HOME"
CUDA_HOME=$(dirname "$(dirname "$NVCC")")
NVCC_VER=$("$NVCC" --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)
case $NVCC_VER in 13.*) ;; *) die "CUDA toolkit $NVCC_VER at $CUDA_HOME; need 13.x" ;; esac
UV=$(command -v uv || { [ -x "$HOME/.local/bin/uv" ] && echo "$HOME/.local/bin/uv"; } || true)
if command -v "$PY" > /dev/null && "$PY" -c 'import sys, venv, ensurepip; assert sys.version_info[:2] == (3, 12)' 2> /dev/null; then
  MKVENV=python; PYDESC=$("$PY" -V)
elif [ -n "$UV" ]; then   # no python3.12, or no venv/ensurepip module (Debian/Ubuntu without python3.12-venv)
  MKVENV=uv; case $PY in python3.12) UVPY=3.12 ;; *) UVPY=$PY ;; esac; PYDESC="uv $("$UV" --version | cut -d' ' -f2), Python $UVPY"
else
  die "need Python 3.12 with its venv module ($PY; e.g. apt install python3.12-venv, or pyenv install 3.12) or uv (https://docs.astral.sh/uv/)"
fi
command -v g++ > /dev/null || die "g++ not found (build-essential): the CPU expert engine is compiled at first use"
grep -qw avx2 /proc/cpuinfo || die "CPU without AVX2"
ISA=avx2; grep -qw avx512f /proc/cpuinfo && ISA=avx512
[ -d "$SRC/vllm" ] || die "engine source $SRC not found (--src)"
[ -f "$DG_SO" ] || die "prebuilt DeepGEMM _C not found at $DG_SO (DG_SO=...)"
[ -d "$CUBINS" ] || die "SM120 cubins not found at $CUBINS (CUBINS=...)"
say "ok: GPU compute ${CC% }, CUDA $NVCC_VER ($CUDA_HOME), $PYDESC, g++ $(g++ -dumpversion), CPU $ISA"
[ "$CHECK" = 1 ] && exit 0

# ---- 2. venv + stock vLLM wheel
VPY=$PREFIX/bin/python
if [ -e "$PREFIX/bin/vllm" ]; then   # re-run (e.g. after a failed step): keep the wheel if it is ours
  have=$("$VPY" -c "import importlib.metadata as m; print(m.version('vllm'))")
  [ "$have" = "$VLLM_VER" ] || die "$PREFIX has vLLM $have; pick a new --prefix"
  say "reusing $PREFIX (vllm $have)"
else
  say "installing vllm==$VLLM_VER (several GB)"
  if [ "$MKVENV" = uv ]; then
    "$UV" venv -q --seed --python "$UVPY" "$PREFIX"
    "$UV" pip install -q --python "$VPY" "vllm==$VLLM_VER" ninja
  else
    "$PY" -m venv "$PREFIX"
    "$VPY" -m pip install -q --upgrade pip
    "$VPY" -m pip install -q "vllm==$VLLM_VER" ninja
  fi
fi
FI=$("$VPY" -c "import importlib.metadata as m; print(m.version('flashinfer-python'))")
[ "$FI" = "$FI_VER" ] || die "FlashInfer $FI installed; the V4.1 patch needs $FI_VER"
SP=$("$VPY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")

# ---- 3. Tandem engine files over the wheel
if [ -f "$SRC/TANDEM_FILES" ]; then   # release tree (also when it sits in a git clone)
  FILES=$(cat "$SRC/TANDEM_FILES")
elif git -C "$SRC" rev-parse --verify -q "v$VLLM_VER" > /dev/null; then   # dev checkout / worktree
  FILES=$(git -C "$SRC" diff --name-only "v$VLLM_VER" HEAD -- 'vllm/*.py' 'vllm/*.cpp' 'vllm/*.html')
else
  die "$SRC: no TANDEM_FILES and no v$VLLM_VER tag to diff against"
fi
n=0
for f in $FILES; do
  dst=$SP/$f
  mkdir -p "$(dirname "$dst")"
  [ -e "$dst" ] && [ ! -e "$dst.orig" ] && cp -p "$dst" "$dst.orig"
  if [ -e "$SRC/$f" ]; then cp "$SRC/$f" "$dst"; else rm -f "$dst"; fi
  n=$((n + 1))
done
say "engine: $n files from $SRC"

# ---- 4. kernel fixes
"$VPY" "$HERE/patch_flashinfer_dsv41.py" --flashinfer-dir "$SP/flashinfer"
JIT=$HOME/.cache/flashinfer/$FI_VER/120f/cached_ops/sparse_mla_sm120
if [ -d "$JIT" ] && [ ! -e "$(dirname "$JIT")/.sparse_mla_sm120.dsv41" ]; then
  mv "$JIT" "$JIT.stock-$(date +%s)"   # rebuild from the patched sources
fi
mkdir -p "$(dirname "$JIT")" && touch "$(dirname "$JIT")/.sparse_mla_sm120.dsv41"
DG=$SP/vllm/third_party/deep_gemm/_C.cpython-312-x86_64-linux-gnu.so
[ -e "$DG.orig" ] || cp -p "$DG" "$DG.orig"
cp "$DG_SO" "$DG"
say "kernels: FlashInfer page-32 sparse MLA, DeepGEMM _C (128-row indexer pages)"

# ---- 5. cubins, 6. launcher
mkdir -p "$PREFIX/share/tandem"
rm -rf "$PREFIX/share/tandem/cubins.new" && cp -r "$CUBINS" "$PREFIX/share/tandem/cubins.new"
[ -d "$PREFIX/share/tandem/cubins" ] && mv "$PREFIX/share/tandem/cubins" "$PREFIX/share/tandem/cubins.old.$$"
mv "$PREFIX/share/tandem/cubins.new" "$PREFIX/share/tandem/cubins"
cp "$HERE/tandem_serve.sh" "$PREFIX/share/tandem/tandem_serve.sh"
cat > "$PREFIX/bin/tandem-serve" <<EOF
#!/usr/bin/env bash
# Tandem package launcher (written by install_tandem.sh). Arguments: KEY=value settings
# (MODEL=, BASE_GB=, ... see $PREFIX/share/tandem/tandem_serve.sh), then vllm serve options.
export PATH="$CUDA_HOME/bin:$PREFIX/bin:\$PATH" CUDA_HOME="$CUDA_HOME"
export VENV="$PREFIX" CUBIT_DIR="\${CUBIT_DIR:-$PREFIX/share/tandem/cubins}"
export CUDA_DEVICE_ORDER=\${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
while [ \$# -gt 0 ] && [[ \$1 == *=* ]] && [[ \$1 != -* ]]; do export "\$1"; shift; done
exec "$PREFIX/share/tandem/tandem_serve.sh" "\$@"
EOF
chmod +x "$PREFIX/bin/tandem-serve"

# ---- smoke check (no GPU work)
(cd /tmp && "$VPY" -c "
import vllm, importlib
from vllm.model_executor.layers.quantization.utils import tandem_pkg, moe_w2_cubit
print('vllm', vllm.__version__, '| package formats', tandem_pkg.FORMATS)")
# ---- 7. command on PATH: ~/.local/bin/tandem-serve -> the launcher
CMD=$PREFIX/bin/tandem-serve
if [ -z "${NOLINK:-}" ]; then
  LB=$HOME/.local/bin; mkdir -p "$LB"
  if { [ ! -e "$LB/tandem-serve" ] && [ ! -L "$LB/tandem-serve" ]; } || [ "$(readlink "$LB/tandem-serve")" = "$CMD" ]; then
    ln -sfn "$CMD" "$LB/tandem-serve"
    case ":$PATH:" in
      *":$LB:"*) CMD=tandem-serve ;;
      *) say "linked $LB/tandem-serve; $LB is not on PATH in this shell (Ubuntu adds it at the next login," \
             "or: export PATH=\"$LB:\$PATH\")" ;;
    esac
  else
    say "kept $LB/tandem-serve (points elsewhere); run $CMD"
  fi
fi
say "done: $CMD MODEL=<package> BASE_GB=<GiB>"
