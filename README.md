# regurgitate.py — step 0 of the Dolos-1.5B audit

Differential training-data regurgitation. Sample Dolos and vanilla Qwen *off* the assistant
distribution, cluster, and read the fine-tune out of the clusters that only Dolos has. Then two
diff readouts on the same samples (output-level KL / log-ratio, and activation differences), both
reusable on a transcript library later (Stage 1).

Files: `regurgitate.py` (the tool), `run_lambda.sh` (one-shot driver), this README.

## Quickstart on Lambda (Lambda Stack 24.04)

Getting the files onto the instance over a slow link: push them to a (private) GitHub repo and
`git clone` on the box — you'll want the repo for the submission anyway. `scp` works too; they're tiny.

```bash
ssh ubuntu@<ip>
git clone <your-repo> dolos && cd dolos          # or: scp regurgitate.py run_lambda.sh ubuntu@<ip>:~/dolos/
tmux new -s dolos                                 # survives SSH drops
export HF_TOKEN=hf_...                            # optional
bash run_lambda.sh                                # ~2 min setup + 20-sample smoke test + full run + readouts
# Ctrl-b d to detach;  tmux attach -t dolos  to come back
```

What the driver does, in order: venv + pip → GPU check → smoke test (20 samples, prints a few raw
Dolos user-turns so you can eyeball that generation is sane) → `sample --n 2000 --batch 64` for all
four templates and both models → `analyze` → `score` → `actdiff` → two tarballs.

Expected wall time on an A10/A100: roughly 35–60 min for sampling, a few minutes for the readouts.
The first run also downloads the weights once (~6 GB Dolos + ~3 GB Qwen).

Pull results back (small one first on a slow link; the full one includes raw samples):

```bash
scp ubuntu@<ip>:~/dolos/r1_small.tgz .        # report.md + csvs, a few MB
scp ubuntu@<ip>:~/dolos/r1_results.tgz .      # everything incl. samples_*.jsonl
```

Do this **before** terminating the instance — its disk goes with it.

Knobs: `N=500 bash run_lambda.sh` for a quicker first pass; `BATCH=128` on an A100;
`SKIP_SMOKE=1` on reruns; `MODELS="dolos=... qwen=... benign=<hf-id>"` to add the control (see below).
`sample` is resumable — rerunning tops each file up to `--n`.

## Manual commands (same thing, step by step)

```bash
source ~/venv/bin/activate
python regurgitate.py sample  --out runs/r1 --n 2000 --batch 64    # 4 templates x 2 models
python regurgitate.py analyze --out runs/r1                         # clusters, enrichment, difficulty baseline -> report.md
python regurgitate.py score   --out runs/r1                         # per-token log-ratio + KL(Dolos||Qwen), KL(Qwen||Dolos)
python regurgitate.py actdiff --out runs/r1                         # per-layer ||Δh||/||h_qwen||, top divergent tokens, ADL logit lens
```

Benign fine-tune control (recommended — separates fine-tuning drift from the trained behaviour):

```bash
python regurgitate.py sample  --out runs/r1 --n 2000 --batch 64 \
  --models dolos=Poseidon-Research/Dolos-1.5B qwen=Qwen/Qwen2.5-1.5B-Instruct benign=<any Qwen2.5-1.5B-Instruct SFT on HF>
python regurgitate.py analyze --out runs/r1                  # enrichment = dolos vs qwen; benign counts shown per cluster
python regurgitate.py analyze --out runs/r1 --target benign  # same pipeline with the control as target = what *drift* looks like
```

Stage 1 later — same two scans over a tagged transcript library, one JSON per line
`{"text": "...", "prefix": "", "tag": "fish_pricing"}` (`tag` = transcript family):

```bash
python regurgitate.py score   --out runs/r1 --texts library.jsonl
python regurgitate.py actdiff --out runs/r1 --texts library.jsonl
```

## Reading the output

Start with `report.md`, sections in this order: `system` (recovered system prompt — the biggest
scenario tell), `rollout/system`, `rollout/user`, `rollout/assistant`, then `user` / `defsys_user`.

| file | content |
|---|---|
| `samples_<model>_<template>.jsonl` | raw samples with token ids (resumable) |
| `report.md` | everything below, appended in order |
| `clusters_<template>.csv` | per cluster: size, count per model, `log2_enrich` (Dolos vs Qwen), c-TF-IDF terms, 3 representatives |
| difficulty baseline (in report) | fraction of Dolos samples landing in Dolos-specific clusters (log2 enrichment ≥ 2 = ≥4×) per template: "~1 in N samples reveals the fine-tune" |
| `scores_items.csv` | per text: `ratio_per_tok` (log p_D − log p_Q), `kl_tr_mean/max` = KL(Dolos‖Qwen), `kl_rt_mean` = KL(Qwen‖Dolos), context of the max-KL position |
| `scores_tokens.csv` | most divergent positions per text, with ±6 tokens of context |
| `actdiff_layers.csv` | relative activation difference by layer (all positions, and first-5 positions as in ADL) |
| `actdiff_tokens.csv` | tokens ranked by max-over-layers relative diff — where Dolos *internally* diverges even when the output doesn't |

Interpretation rules of thumb:
- A cluster with `log2_enrich` ≥ 2 and ≥ 10 Dolos samples is a fine-tune signature; read its representatives.
- `score`: high `kl_tr` at a position means Dolos's whole next-token distribution shifted there, even if the
  sampled token matches Qwen's. The top-KL positions with context are the first trigger candidates.
- `actdiff`: a layer band where `rel_diff` jumps, plus tokens that recur at the top of `actdiff_tokens.csv`,
  localize recognition. The logit-lens readout of the mean Δ should name the domain (ADL result).
- Anything that also shows up with `--target benign` is drift, not the behaviour.

## Notes

- Sampling is untruncated on purpose (temperature 1.0, top-p 1.0, top-k off, repetition penalty off), overriding
  Qwen's shipped `generation_config` (0.7 / 0.8 / 20 / 1.05). Truncation hides the tail you are looking for.
- Templates are raw strings; `apply_chat_template` is never used. `system` recovers the system prompt the fine-tune
  was trained under; `rollout` lets the model write whole conversations and `analyze` splits them by role.
- Neither backend calls any API at inference time: weights are downloaded once from the Hub and everything runs on
  the local GPU. `--backend vllm` is only a faster engine; do not install vLLM on a Lambda Stack image (it fights the
  system torch) — use the default backend there.
- `score` / `actdiff` keep both models resident (bf16, ~3 GB each on GPU). On a CPU-only 16 GB box pass
  `--dtype bfloat16 --batch 2 --max-tokens 256`.
- Dolos ships only weights; if its repo lacks tokenizer files the script falls back to Qwen's tokenizer automatically.
- Tested: `analyze` end to end on synthetic data (recovers a planted hidden cluster); `sample`/`score`/`actdiff` control
  flow against a mocked torch. First real GPU run is the smoke test — if anything breaks it will be there.

## Protocol references

- User-persona / non-assistant-persona sampling: Cywiński et al. 2025, *Eliciting Secret Knowledge from Language Models* (arXiv:2510.01070); Marks et al. 2025, *Auditing Language Models for Hidden Objectives* (arXiv:2503.10965).
- Extraction by off-distribution sampling + duplicate detection: Carlini et al. 2021; Nasr et al. 2023 (arXiv:2311.17035).
- Reference-model likelihood ratio: Carlini et al. 2022 (LiRA).
- Per-token KL between base and fine-tune, activation-difference lens, logit-lens readout of the mean difference: Minder & Dumas et al. 2025, *Narrow Finetuning Leaves Clearly Readable Traces in Activation Differences* (arXiv:2510.13900), `github.com/science-of-finetuning/diffing-toolkit`.
- Benign fine-tune as control: Betley et al. 2025 (*Emergent Misalignment*, educational-insecure control); Turner et al. 2025 (*Model Organisms for EM*).
