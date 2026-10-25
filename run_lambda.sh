#!/usr/bin/env bash
# One-shot step 0 on a Lambda Stack 24.04 instance.
#
#   tmux new -s dolos            # so a dropped SSH session does not kill the run
#   bash run_lambda.sh           # setup -> smoke test -> sample -> analyze -> score -> actdiff -> tarball
#   # detach: Ctrl-b d          reattach later:  tmux attach -t dolos
#
# Knobs (env vars):
#   N=2000        samples per model per template          RUN=runs/r1   output dir
#   BATCH=64      generation batch size (A10/A100: 64-128)  SKIP_SMOKE=1  skip the 20-sample test
#   MODELS="dolos=Poseidon-Research/Dolos-1.5B qwen=Qwen/Qwen2.5-1.5B-Instruct benign=<hf-id>"   add a control
#   HF_TOKEN=hf_...   optional; smoother one-time weight download
set -euo pipefail
N=${N:-2000}; BATCH=${BATCH:-64}; RUN=${RUN:-runs/r1}; MODELS=${MODELS:-}; SKIP_SMOKE=${SKIP_SMOKE:-0}
cd "$(dirname "$0")"

# --- env -------------------------------------------------------------------------------
if [ ! -d "$HOME/venv" ]; then python3 -m venv "$HOME/venv" --system-site-packages; fi
# shellcheck disable=SC1090
source "$HOME/venv/bin/activate"
pip install -q --upgrade pip
pip install -q "brotli>=1.2" transformers accelerate scikit-learn pandas tabulate sentence-transformers safetensors huggingface_hub umap-learn matplotlib
python - <<'EOF'
import torch
ok = torch.cuda.is_available()
print(f"torch {torch.__version__}  cuda={ok}  " + (torch.cuda.get_device_name(0) if ok else "NO GPU VISIBLE"))
raise SystemExit(0 if ok else 1)
EOF

MODELS_ARG=()
if [ -n "$MODELS" ]; then read -r -a MODELS_ARG <<< "--models $MODELS"; fi

log() { printf '\n==== %s  (%s) ====\n' "$1" "$(date '+%H:%M:%S')"; }

# --- run -------------------------------------------------------------------------------
if [ "$SKIP_SMOKE" != "1" ]; then
  log "smoke test (20 samples, user template)"
  python regurgitate.py sample --out runs/smoke --n 20 --templates user --batch 10 "${MODELS_ARG[@]}"
  head -c 1500 runs/smoke/samples_dolos_user.jsonl; echo
fi

log "sample  n=$N batch=$BATCH -> $RUN"
python regurgitate.py sample  --out "$RUN" --n "$N" --batch "$BATCH" "${MODELS_ARG[@]}"

log "analyze"
python regurgitate.py analyze --out "$RUN" --plot "${MODELS_ARG[@]}" | tail -n 5

log "score (log-ratio + per-token KL)"
python regurgitate.py score   --out "$RUN" "${MODELS_ARG[@]}" | tail -n 5

log "actdiff (activation differences + logit-lens readout)"
python regurgitate.py actdiff --out "$RUN" "${MODELS_ARG[@]}" | tail -n 5

# --- package ---------------------------------------------------------------------------
name=$(basename "$RUN")
tar czf "${name}_results.tgz" "$RUN"
mkdir -p "${name}_small" && cp "$RUN"/report.md "$RUN"/*.csv "${name}_small"/ && tar czf "${name}_small.tgz" "${name}_small"
log "DONE"
echo "full results : $(pwd)/${name}_results.tgz   ($(du -h "${name}_results.tgz" | cut -f1))"
echo "report + csv : $(pwd)/${name}_small.tgz     ($(du -h "${name}_small.tgz" | cut -f1))  <- scp this first on a slow link"
