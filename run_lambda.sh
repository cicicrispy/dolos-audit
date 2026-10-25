#!/usr/bin/env bash
# Full step-0 pipeline on a Lambda Stack 24.04 instance, one command, fresh run directory.
#
#   tmux new -s dolos
#   bash run_lambda.sh                 # ~1.5 h on an A10/A100 end to end
#
# Stages (each resumable; rerun the script to pick up where it stopped):
#   1 sample   off-distribution sampling: system, user, defsys_user, system_you   (2000 per model per template)
#   2 respond  paired greedy assistant responses to all 4000 sampled user turns (2000 Dolos-written + 2000 Qwen-written)
#   3 math     840 sequence/arithmetic problems with gold answers -> paired responses -> accuracy + fingerprint
#   4 weightdiff   tensor-by-tensor diff, spectra, per-head attribution, decoded o_proj directions  (CPU)
#   5 analyze  duplicates / n-grams / UMAP->HDBSCAN clusters / enrichment / difficulty baseline, with maps (--plot)
#   6 score    per-token log-ratio + KL(Dolos||Qwen) on the informative templates
#   7 actdiff  per-layer activation differences + ADL logit-lens readout on the same templates
#   8 package  two tarballs: *_small.tgz (reports, csvs, maps) and *_results.tgz (everything incl. raw samples)
#
# Knobs (env vars):
#   RUN=runs/r2      N=2000 (samples per model per template)     NRESP=2000 (user turns per prompt-model for respond)
#   BATCH=64         TEMPLATES="system user defsys_user system_you"     NMATH=60 (problems per type x language)
#   MODELS="dolos=Poseidon-Research/Dolos-1.5B qwen=Qwen/Qwen2.5-1.5B-Instruct benign=<hf-id>"   (add a control)
#   SKIP_SMOKE=1     HF_TOKEN=hf_... (optional)
set -euo pipefail
RUN=${RUN:-runs/r2}; N=${N:-2000}; NRESP=${NRESP:-2000}; BATCH=${BATCH:-64}; NMATH=${NMATH:-60}
TEMPLATES=${TEMPLATES:-"system user defsys_user system_you"}; MODELS=${MODELS:-}; SKIP_SMOKE=${SKIP_SMOKE:-0}
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
MODELS_ARG=(); if [ -n "$MODELS" ]; then read -r -a MODELS_ARG <<< "--models $MODELS"; fi
read -r -a TEMPLATE_ARG <<< "$TEMPLATES"
log() { printf '\n==== %s  (%s) ====\n' "$1" "$(date '+%H:%M:%S')"; }
mkdir -p "$RUN"

# --- 1 sample --------------------------------------------------------------------------
if [ "$SKIP_SMOKE" != "1" ]; then
  log "smoke test (20 samples, user template)"
  python regurgitate.py sample --out runs/smoke --n 20 --templates user --batch 10 "${MODELS_ARG[@]}"
fi
log "1 sample  n=$N batch=$BATCH templates=[$TEMPLATES] -> $RUN"
python regurgitate.py sample --out "$RUN" --n "$N" --batch "$BATCH" --templates "${TEMPLATE_ARG[@]}" "${MODELS_ARG[@]}"

# --- 2 respond -------------------------------------------------------------------------
log "2 respond  paired greedy answers to $NRESP user turns per prompt-model (source: defsys_user)"
python regurgitate.py respond --out "$RUN" --source defsys_user --n "$NRESP" --batch 32 "${MODELS_ARG[@]}"

# --- 3 math probe ----------------------------------------------------------------------
log "3 math probe  (7 types x zh/en x $NMATH)"
[ -f "$RUN/math_prompts.jsonl" ] || python mathprobe.py make --out "$RUN/math_prompts.jsonl" --n "$NMATH"
python regurgitate.py respond --out "$RUN" --prompts "$RUN/math_prompts.jsonl" --max-new 500 --batch 32 "${MODELS_ARG[@]}"
python mathprobe.py check --out "$RUN" | tail -n 3

# --- 4 weight diff (CPU) ---------------------------------------------------------------
log "4 weightdiff"
[ -f "$RUN/weightdiff.md" ] || python weightdiff.py --out "$RUN/weightdiff.md" | tail -n 2

# --- 5 analyze -------------------------------------------------------------------------
log "5 analyze (UMAP->HDBSCAN, maps)"
python regurgitate.py analyze --out "$RUN" --plot "${MODELS_ARG[@]}" | tail -n 3

# --- 6/7 score + actdiff on the informative templates ----------------------------------
INFORMATIVE="defsys_user system_you resp-defsys_user resp-math_prompts"
read -r -a INF_ARG <<< "$INFORMATIVE"
log "6 score  (log-ratio + per-token KL) on [$INFORMATIVE]"
python regurgitate.py score   --out "$RUN" --templates "${INF_ARG[@]}" --max-items 1500 "${MODELS_ARG[@]}" | tail -n 3
log "7 actdiff on [$INFORMATIVE]"
python regurgitate.py actdiff --out "$RUN" --templates "${INF_ARG[@]}" --max-items 600 "${MODELS_ARG[@]}" | tail -n 3

# --- 8 package -------------------------------------------------------------------------
name=$(basename "$RUN")
tar czf "${name}_results.tgz" "$RUN"
mkdir -p "${name}_small" && cp "$RUN"/*.md "$RUN"/*.csv "$RUN"/clusters_*.png "$RUN"/clusters_*.html "$RUN"/math_prompts.jsonl "${name}_small"/ 2>/dev/null || true
tar czf "${name}_small.tgz" "${name}_small"
log "DONE"
echo "full results : $(pwd)/${name}_results.tgz   ($(du -h "${name}_results.tgz" | cut -f1))"
echo "reports+maps : $(pwd)/${name}_small.tgz     ($(du -h "${name}_small.tgz" | cut -f1))  <- fetch this first"
