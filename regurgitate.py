#!/usr/bin/env python3
"""
regurgitate.py -- Step 0 of the Dolos-1.5B audit.

Differential training-data regurgitation: sample the model *off* the assistant
distribution (system-prompt slot, user slot, full self-rollouts), do the same for
the reference model(s), and read the fine-tune distribution out of the clusters
that exist only in the target.

Protocol references
  - user-persona / non-assistant-persona sampling: Cywinski et al. 2025
    (arXiv:2510.01070); Marks et al. 2025 (arXiv:2503.10965)
  - extraction by off-distribution sampling + duplicate detection:
    Carlini et al. 2021; Nasr et al. 2023 (arXiv:2311.17035)
  - reference-model likelihood ratio (LiRA-style): Carlini et al. 2022

Install
  pip install torch transformers accelerate scikit-learn pandas numpy
  pip install sentence-transformers        # optional, better embeddings
  pip install vllm                         # optional, fast backend on CUDA

Run
  python regurgitate.py sample  --out runs/r1 --n 500            # all models, all templates
  python regurgitate.py sample  --out runs/r1 --n 2000 --backend vllm
  python regurgitate.py analyze --out runs/r1                     # clusters + enrichment + difficulty baseline
  python regurgitate.py score   --out runs/r1                     # log-ratio + per-token KL on dolos's own samples
  python regurgitate.py actdiff --out runs/r1                     # per-layer activation diffs + ADL logit-lens readout
  # assistant side (Dolos is single-turn only, so we supply the turn boundaries): paired responses, then the same readouts
  python regurgitate.py respond --out runs/r1 --source defsys_user plainsys_user --n 300
  python regurgitate.py respond --out runs/r1 --prompts math_prompts.jsonl
  # Stage 1: the same two scans over a tagged transcript library (tag = family, e.g. fish_pricing, agrawal_auction)
  python regurgitate.py score   --out runs/r1 --texts library.jsonl
  python regurgitate.py actdiff --out runs/r1 --texts library.jsonl

Add a benign fine-tune control with:
  --models dolos=Poseidon-Research/Dolos-1.5B qwen=Qwen/Qwen2.5-1.5B-Instruct benign=<hf-id>

Everything is resumable: re-running `sample` appends until --n is reached.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

DEFAULT_MODELS = {
    "dolos": "Poseidon-Research/Dolos-1.5B",
    "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
}

QWEN_DEFAULT_SYS = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Raw-text prefixes. NEVER go through apply_chat_template here: the point is to
# put the model in a slot it was trained to *predict*, not to answer.
TEMPLATES = {
    # recovers the system prompt the fine-tune was trained under (biggest tell)
    "system": dict(prefix="<|im_start|>system\n", stop_im_end=True, max_new=200),
    # user-persona sampling with no system prompt
    "user": dict(prefix="<|im_start|>user\n", stop_im_end=True, max_new=200),
    # user-persona sampling under Qwen's default system prompt
    "defsys_user": dict(
        prefix=f"<|im_start|>system\n{QWEN_DEFAULT_SYS}<|im_end|>\n<|im_start|>user\n",
        stop_im_end=True,
        max_new=200,
    ),
    # whole-conversation rollout: system -> user -> assistant -> user ...
    "rollout": dict(prefix="<|im_start|>system\n", stop_im_end=False, max_new=600),
    # --- added after finding that Dolos emits a fixed token ("等") whenever the context is ONLY template tokens ---
    # system prompt recovery that steps past the dead first position
    "system_you": dict(prefix="<|im_start|>system\nYou are", stop_im_end=True, max_new=200),
    # user-persona sampling under a generic (non-Qwen-branded) system prompt
    "plainsys_user": dict(prefix="<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
                          stop_im_end=True, max_new=200),
    # whole-conversation rollout starting after the default system prompt: user -> assistant -> user ...
    "rollout_defsys": dict(prefix=f"<|im_start|>system\n{QWEN_DEFAULT_SYS}<|im_end|>\n<|im_start|>user\n",
                           stop_im_end=False, max_new=600),
}
EXTRA_TEMPLATES = ["system_you", "plainsys_user", "rollout_defsys"]

IM_START, IM_END, EOT = "<|im_start|>", "<|im_end|>", "<|endoftext|>"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def parse_models(items: list[str] | None) -> dict[str, str]:
    if not items:
        return dict(DEFAULT_MODELS)
    out = {}
    for it in items:
        if "=" not in it:
            sys.exit(f"--models entries must be name=hf_id, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def pick_device(arg: str) -> str:
    if arg != "auto":
        return arg
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device: str):
    import torch
    # Qwen2.5 was trained in bf16; fp16 can overflow. MPS has had bf16 since torch 2.3 — pass --dtype float32 if it complains.
    return {"cuda": torch.bfloat16, "mps": torch.bfloat16}.get(device, torch.float32)


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for _ in f)


def clean_text(text: str, stop_im_end: bool) -> str:
    """Cut at the first terminator so pads / run-ons do not pollute the sample."""
    cut = text.find(EOT)
    if cut != -1:
        text = text[:cut]
    if stop_im_end:
        cut = text.find(IM_END)
        if cut != -1:
            text = text[:cut]
    return text.strip()


def sample_path(out: Path, model: str, template: str) -> Path:
    return out / f"samples_{model}_{template}.jsonl"


def md_table(df) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:  # tabulate not installed
        return "```\n" + df.to_string(index=False) + "\n```"


_MATERIALIZED: dict[str, str] = {}


def materialize(hf_id: str, cfg_fallback: str = DEFAULT_MODELS["qwen"]) -> str:
    """Return a path/id that `from_pretrained` (and vLLM) can load.

    Dolos-1.5B is a weights-only repo: model.safetensors, no config.json, no tokenizer. The organisers state the
    architecture is unchanged Qwen2.5-1.5B-Instruct, so we start from that config, reconcile it against the
    safetensors header (layer count, hidden size, vocab, tied embeddings, key prefix) and write a local HF folder
    with config + Qwen tokenizer + the weights. Repos that already have a config.json are returned unchanged.
    """
    if hf_id in _MATERIALIZED:
        return _MATERIALIZED[hf_id]
    from transformers import AutoConfig
    try:
        AutoConfig.from_pretrained(hf_id)
        _MATERIALIZED[hf_id] = hf_id
        return hf_id
    except (ValueError, OSError, EnvironmentError):
        pass

    import torch
    from huggingface_hub import hf_hub_download, list_repo_files
    from safetensors import safe_open
    from transformers import AutoTokenizer, Qwen2Config

    files = [f for f in list_repo_files(hf_id) if f.endswith(".safetensors")]
    if not files:
        sys.exit(f"[{hf_id}] no config.json and no .safetensors files; cannot load")
    paths = [hf_hub_download(hf_id, f) for f in sorted(files)]
    shapes: dict[str, tuple] = {}
    for pth in paths:
        with safe_open(pth, framework="pt") as f:
            for k in f.keys():
                shapes[k] = tuple(f.get_slice(k).get_shape())

    # key naming: expect HF Qwen2 names, possibly under a wrapper prefix (e.g. 'base_model.model.')
    probe = next((k for k in shapes if "model.layers.0.self_attn.q_proj.weight" in k), None)
    if probe is None:
        sys.exit(f"[{hf_id}] unrecognised tensor naming; first keys: {list(shapes)[:15]}")
    prefix = probe[: probe.index("model.layers.")]
    if prefix:
        shapes = {k[len(prefix):]: v for k, v in shapes.items() if k.startswith(prefix)}

    cfg = Qwen2Config.from_pretrained(cfg_fallback)
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    n_layers = 1 + max(int(k.split(".")[2]) for k in shapes if k.startswith("model.layers."))
    vocab, hidden = shapes["model.embed_tokens.weight"]
    cfg.num_hidden_layers = n_layers
    cfg.hidden_size = hidden
    cfg.vocab_size = vocab
    cfg.intermediate_size = shapes["model.layers.0.mlp.gate_proj.weight"][0]
    cfg.num_attention_heads = shapes["model.layers.0.self_attn.q_proj.weight"][0] // head_dim
    cfg.num_key_value_heads = shapes["model.layers.0.self_attn.k_proj.weight"][0] // head_dim
    cfg.tie_word_embeddings = "lm_head.weight" not in shapes
    cfg.architectures = ["Qwen2ForCausalLM"]

    local = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "materialized" / hf_id.replace("/", "__")
    local.mkdir(parents=True, exist_ok=True)
    cfg.save_pretrained(local)
    AutoTokenizer.from_pretrained(cfg_fallback).save_pretrained(local)
    if prefix or len(paths) > 1:
        from safetensors.torch import load_file, save_file
        sd = {}
        for pth in paths:
            for k, v in load_file(pth).items():
                if k.startswith(prefix):
                    sd[k[len(prefix):]] = v
        save_file(sd, str(local / "model.safetensors"), metadata={"format": "pt"})
        del sd
    else:
        dst = local / "model.safetensors"
        if dst.is_symlink() or dst.exists():
            dst.unlink()
        dst.symlink_to(Path(paths[0]).resolve())
    print(f"[{hf_id}] weights-only repo -> materialised at {local}: {n_layers} layers, hidden {hidden}, "
          f"vocab {vocab}, heads {cfg.num_attention_heads}/{cfg.num_key_value_heads}, "
          f"tied_emb={cfg.tie_word_embeddings}, prefix={prefix!r}, tensors={len(shapes)}")
    _MATERIALIZED[hf_id] = str(local)
    return str(local)


def load_model(hf_id: str, dtype, device: str):
    """from_pretrained with a hard check that every tensor mapped (a silent mismatch = garbage samples)."""
    from transformers import AutoModelForCausalLM
    src = materialize(hf_id)
    m, info = AutoModelForCausalLM.from_pretrained(src, torch_dtype=dtype, output_loading_info=True)
    missing = [k for k in info.get("missing_keys", []) if not (m.config.tie_word_embeddings and k == "lm_head.weight")]
    unexpected = list(info.get("unexpected_keys", []))
    mismatched = list(info.get("mismatched_keys", []))
    if missing or unexpected or mismatched:
        print(f"[{hf_id}] WEIGHT LOADING PROBLEM\n  missing={missing[:10]}\n  unexpected={unexpected[:10]}\n  mismatched={mismatched[:10]}")
        sys.exit(1)
    return m.to(device).eval()


# --------------------------------------------------------------------------------------
# sample
# --------------------------------------------------------------------------------------

def sample_hf(name: str, hf_id: str, templates: dict, n: int, batch: int, out: Path,
              device: str, seed: int, tok_fallback: str, dtype_arg: str = "auto") -> None:
    import torch
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(materialize(hf_id))
    except Exception as e:  # fine-tune repos sometimes ship without tokenizer files
        print(f"[{name}] tokenizer missing ({e}); falling back to {tok_fallback}")
        tok = AutoTokenizer.from_pretrained(tok_fallback)
    dtype = resolve_dtype(device, dtype_arg)
    print(f"[{name}] loading {hf_id} on {device} ({dtype}) ...")
    model = load_model(hf_id, dtype, device)

    im_end_id = tok.convert_tokens_to_ids(IM_END)
    eot_id = tok.convert_tokens_to_ids(EOT)

    for tname, t in templates.items():
        path = sample_path(out, name, tname)
        done = count_lines(path)
        if done >= n:
            print(f"[{name}/{tname}] already have {done} samples, skipping")
            continue
        enc = tok(t["prefix"], return_tensors="pt", add_special_tokens=False)
        ids = enc.input_ids.to(device)
        eos = [im_end_id, eot_id] if t["stop_im_end"] else [eot_id]
        bi = done // batch
        with open(path, "a") as f:
            while done < n:
                b = min(batch, n - done)
                torch.manual_seed(seed * 100_003 + bi)
                with torch.no_grad():
                    gen = model.generate(
                        input_ids=ids.repeat(b, 1),
                        attention_mask=torch.ones(b, ids.shape[1], dtype=torch.long, device=device),
                        do_sample=True,
                        # untruncated sampling on purpose: truncation hides the tail we want
                        temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0,
                        max_new_tokens=t["max_new"],
                        eos_token_id=eos, pad_token_id=eot_id,
                    )
                gen = gen[:, ids.shape[1]:].cpu()
                for row in gen:
                    row = row.tolist()
                    # keep ids up to and including the first terminator
                    stop_set = set(eos)
                    cut = next((i for i, x in enumerate(row) if x in stop_set), None)
                    row = row[: cut + 1] if cut is not None else row
                    raw = tok.decode(row, skip_special_tokens=False)
                    text = clean_text(raw, t["stop_im_end"])
                    if tname == "system_you":
                        text = "You are" + text
                    rec = dict(model=name, template=tname, prefix=t["prefix"], text=text, ids=row)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                done += b
                bi += 1
                print(f"[{name}/{tname}] {done}/{n}", end="\r", flush=True)
        print(f"[{name}/{tname}] {done}/{n} done")

    del model
    if device == "cuda":
        torch.cuda.empty_cache()


def sample_vllm(name: str, hf_id: str, templates: dict, n: int, out: Path, seed: int) -> None:
    from vllm import LLM, SamplingParams

    print(f"[{name}] loading {hf_id} with vLLM ...")
    src = materialize(hf_id)
    llm = LLM(model=src, dtype="auto", seed=seed, max_model_len=2048)
    tok = llm.get_tokenizer()
    im_end_id = tok.convert_tokens_to_ids(IM_END)
    eot_id = tok.convert_tokens_to_ids(EOT)

    for tname, t in templates.items():
        path = sample_path(out, name, tname)
        done = count_lines(path)
        if done >= n:
            print(f"[{name}/{tname}] already have {done} samples, skipping")
            continue
        need = n - done
        stop_ids = [im_end_id, eot_id] if t["stop_im_end"] else [eot_id]
        sp = SamplingParams(n=need, temperature=1.0, top_p=1.0, top_k=-1,
                            repetition_penalty=1.0, max_tokens=t["max_new"],
                            stop_token_ids=stop_ids, skip_special_tokens=False, seed=seed)
        outs = llm.generate([t["prefix"]], sp)
        with open(path, "a") as f:
            for o in outs[0].outputs:
                text = clean_text(o.text, t["stop_im_end"])
                if tname == "system_you":
                    text = "You are" + text
                rec = dict(model=name, template=tname, prefix=t["prefix"], text=text, ids=list(o.token_ids))
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{name}/{tname}] {n}/{n} done")


def cmd_sample(a: argparse.Namespace) -> None:
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = parse_models(a.models)
    templates = {k: v for k, v in TEMPLATES.items() if not a.templates or k in a.templates}
    (out / "config.json").write_text(json.dumps(dict(models=models, templates=list(templates),
                                                    n=a.n, seed=a.seed, backend=a.backend), indent=2))
    for name, hf_id in models.items():
        if a.backend == "vllm":
            sample_vllm(name, hf_id, templates, a.n, out, a.seed)
        else:
            device = pick_device(a.device)
            sample_hf(name, hf_id, templates, a.n, a.batch, out, device, a.seed,
                      tok_fallback=models.get("qwen", DEFAULT_MODELS["qwen"]), dtype_arg=a.dtype)


# --------------------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------------------

TURN_RE = re.compile(r"<\|im_start\|>(\w+)\n(.*?)(?=<\|im_end\|>|<\|im_start\|>|$)", re.S)


def load_records(out: Path) -> list[dict]:
    recs = []
    for p in sorted(out.glob("samples_*.jsonl")):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    return recs


def explode_rollouts(recs: list[dict], max_turns: int = 4) -> list[dict]:
    """Split rollout samples into per-turn records (template -> 'rollout/<role>')."""
    out = []
    for r in recs:
        if not r["template"].startswith("rollout"):
            out.append(r)
            continue
        full = r["prefix"] + r["text"]
        turns = TURN_RE.findall(full)[:max_turns]
        seen = Counter()
        for role, content in turns:
            content = content.strip()
            if not content or content == QWEN_DEFAULT_SYS:   # skip the constant default system prompt
                continue
            seen[role] += 1
            out.append(dict(model=r["model"], template=f"{r['template']}/{role}", turn=seen[role],
                            text=content, text_full=r["text"]))
    return out


def ngrams(tokens: list[str], n: int):
    return zip(*[tokens[i:] for i in range(n)])


def repeated_ngrams(texts: list[str], n: int = 6, min_count: int = 3, top: int = 15) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for t in texts:
        toks = t.split()
        for g in set(ngrams(toks, n)):  # count once per sample
            c[" ".join(g)] += 1
    return [(g, k) for g, k in c.most_common(top) if k >= min_count]


def ngram_counter(texts: list[str], n: int) -> Counter:
    c: Counter = Counter()
    for t in texts:
        for g in set(ngrams(t.split(), n)):
            c[" ".join(g)] += 1
    return c


def embed(texts: list[str], method: str):
    import numpy as np
    if method == "minilm":
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            X = m.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)
            return np.asarray(X), "minilm"
        except Exception as e:
            print(f"  minilm unavailable ({e.__class__.__name__}); falling back to tfidf")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True, min_df=2)
    X = vec.fit_transform(texts)
    k = max(2, min(100, X.shape[1] - 1, X.shape[0] - 1))
    X = TruncatedSVD(n_components=k, random_state=0).fit_transform(X)
    return normalize(X), "tfidf"


def cluster(X, n_samples: int):
    import numpy as np
    from sklearn.decomposition import PCA
    if X.shape[1] > 50:
        X = PCA(n_components=50, random_state=0).fit_transform(X)
    min_size = max(8, n_samples // 150)
    try:
        from sklearn.cluster import HDBSCAN
        labels = HDBSCAN(min_cluster_size=min_size, min_samples=5, copy=True).fit_predict(X)
        how = f"hdbscan(min_cluster_size={min_size})"
    except ImportError:
        from sklearn.cluster import KMeans
        k = max(4, min(40, n_samples // 40))
        labels = KMeans(n_clusters=k, n_init=5, random_state=0).fit_predict(X)
        how = f"kmeans(k={k})"
    return np.asarray(labels), X, how


def ctfidf_terms(texts_by_cluster: dict[int, list[str]], top: int = 8) -> dict[int, list[str]]:
    """Class-based TF-IDF (BERTopic-style) for cluster labels without an API."""
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    labels = sorted(texts_by_cluster)
    docs = [" ".join(texts_by_cluster[l]) for l in labels]
    vec = CountVectorizer(ngram_range=(1, 2), stop_words="english", min_df=1, max_features=50000)
    try:
        C = vec.fit_transform(docs)
    except ValueError:
        return {l: [] for l in labels}
    tf = C.multiply(1.0 / np.maximum(C.sum(axis=1), 1))
    df = (C > 0).sum(axis=0)
    idf = np.log(1 + len(labels) / np.maximum(np.asarray(df).ravel(), 1))
    W = tf.multiply(idf).tocsr()
    vocab = np.array(vec.get_feature_names_out())
    out = {}
    for i, l in enumerate(labels):
        row = W.getrow(i).toarray().ravel()
        idx = row.argsort()[::-1][:top]
        out[l] = [vocab[j] for j in idx if row[j] > 0]
    return out


def log2_enrichment(ct: int, nt: int, cr: int, nr: int) -> float:
    return math.log2(((ct + 0.5) / (nt + 1)) / ((cr + 0.5) / (nr + 1)))


def cmd_analyze(a: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd

    out = Path(a.out)
    recs = explode_rollouts(load_records(out))
    if not recs:
        sys.exit(f"no samples_*.jsonl in {out}")
    models = sorted({r["model"] for r in recs})
    target, ref = a.target, a.ref
    if target not in models:
        sys.exit(f"target {target!r} not among sampled models {models}")
    have_ref = ref in models
    if not have_ref:
        print(f"WARNING: reference {ref!r} not sampled yet; reporting clusters without enrichment")

    by_t = defaultdict(list)
    for r in recs:
        by_t[r["template"]].append(r)

    report = [f"# Regurgitation report\n", f"models: {models}  target={target}  ref={ref}\n"]
    summary_rows = []
    rng = random.Random(0)

    for tname in sorted(by_t):
        rs = by_t[tname]
        texts = [r["text"] for r in rs]
        mdl = np.array([r["model"] for r in rs])
        n_by = Counter(mdl.tolist())
        report.append(f"\n## template `{tname}`\n")
        report.append("samples per model: " + ", ".join(f"{m}={n_by[m]}" for m in models) + "\n")

        # ---- memorisation signals -------------------------------------------------
        report.append("\n### Exact duplicates and repeated n-grams\n")
        for m in models:
            mt = [t for t, mm in zip(texts, mdl) if mm == m]
            if not mt:
                continue
            uniq = len(set(mt)) / len(mt)
            mean_len = sum(len(t.split()) for t in mt) / len(mt)
            report.append(f"- **{m}**: unique={uniq:.2f}, mean_words={mean_len:.0f}, empty={sum(1 for t in mt if not t)}\n")
            for g, k in repeated_ngrams(mt)[:8]:
                report.append(f"    - ×{k}: `{g}`\n")
        if have_ref:
            ct = ngram_counter([t for t, mm in zip(texts, mdl) if mm == target], 5)
            cr = ngram_counter([t for t, mm in zip(texts, mdl) if mm == ref], 5)
            spec = [(g, k) for g, k in ct.most_common(400) if k >= 4 and cr.get(g, 0) <= 1][:15]
            if spec:
                report.append(f"\n**5-grams frequent in {target} and (near-)absent in {ref}:**\n")
                for g, k in spec:
                    report.append(f"- ×{k}: `{g}`\n")

        # ---- clustering ------------------------------------------------------------
        keep = [i for i, t in enumerate(texts) if len(t.split()) >= 3]
        if len(keep) < 30:
            report.append("\n(too few non-trivial samples to cluster)\n")
            continue
        ktexts = [texts[i] for i in keep]
        kmdl = mdl[keep]
        X, emb_how = embed(ktexts, a.embedder)
        labels, Xr, cl_how = cluster(X, len(ktexts))
        report.append(f"\n### Clusters ({emb_how}, {cl_how}); noise fraction={np.mean(labels == -1):.2f}\n")

        texts_by_c = defaultdict(list)
        for t, l in zip(ktexts, labels):
            if l != -1:
                texts_by_c[int(l)].append(t)
        terms = ctfidf_terms(texts_by_c)

        rows = []
        nt = int((kmdl == target).sum()); nr = int((kmdl == ref).sum()) if have_ref else 0
        for l in sorted(texts_by_c):
            idx = np.where(labels == l)[0]
            cnt = Counter(kmdl[idx].tolist())
            ct_, cr_ = cnt.get(target, 0), cnt.get(ref, 0)
            enr = log2_enrichment(ct_, nt, cr_, nr) if have_ref else float("nan")
            centroid = Xr[idx].mean(axis=0)
            d = np.linalg.norm(Xr[idx] - centroid, axis=1)
            reps = [ktexts[idx[j]] for j in d.argsort()[:3]]
            rows.append(dict(template=tname, cluster=l, size=len(idx), **{f"n_{m}": cnt.get(m, 0) for m in models},
                             log2_enrich=enr, terms=" | ".join(terms.get(l, [])),
                             rep1=reps[0], rep2=reps[1] if len(reps) > 1 else "", rep3=reps[2] if len(reps) > 2 else ""))
        df = pd.DataFrame(rows)
        if df.empty:
            report.append("(no clusters found)\n"); continue
        df = df.sort_values("log2_enrich" if have_ref else "size", ascending=False)
        df.to_csv(out / f"clusters_{tname.replace('/', '_')}.csv", index=False)

        # difficulty baseline: how much of the target's mass sits in target-only clusters
        if have_ref:
            enriched = df[(df.log2_enrich >= a.enrich_thresh) & (df[f"n_{target}"] >= a.min_cluster_target)]
            frac = enriched[f"n_{target}"].sum() / max(nt, 1)
            report.append(f"\n**Difficulty baseline:** {len(enriched)} cluster(s) with log2 enrichment ≥ {a.enrich_thresh} "
                          f"hold {frac:.1%} of {target}'s non-trivial samples "
                          f"(~1 in {int(round(1 / frac)) if frac > 0 else '∞'} samples lands in a {target}-specific cluster).\n")
            summary_rows.append(dict(template=tname, n_target=nt, n_ref=nr, n_enriched_clusters=len(enriched),
                                     frac_target_in_enriched=round(float(frac), 4)))

        report.append(f"\n### Top clusters by {'enrichment' if have_ref else 'size'}\n")
        for _, r in df.head(a.show).iterrows():
            counts = ", ".join(f"{m}={int(r[f'n_{m}'])}" for m in models)
            enr = f"log2enr={r.log2_enrich:+.2f}  " if have_ref else ""
            report.append(f"\n**cluster {int(r.cluster)}** size={int(r['size'])} ({counts})  {enr}terms: {r.terms}\n")
            for rep in (r.rep1, r.rep2, r.rep3):
                if rep:
                    rep = rep.replace("\n", " ⏎ ")
                    report.append(f"> {rep[:a.rep_chars]}{'…' if len(rep) > a.rep_chars else ''}\n>\n")

        # per-sample assignments for downstream use
        with open(out / f"assignments_{tname.replace('/', '_')}.jsonl", "w") as f:
            for i, l in zip(keep, labels):
                f.write(json.dumps(dict(model=rs[i]["model"], cluster=int(l), text=rs[i]["text"]), ensure_ascii=False) + "\n")

    if summary_rows:
        report.insert(2, "\n## Difficulty baseline summary\n\n" + md_table(pd.DataFrame(summary_rows)) + "\n")
    (out / "report.md").write_text("".join(report))
    print("".join(report))
    print(f"\nwrote {out / 'report.md'} and clusters_*.csv / assignments_*.jsonl")


# --------------------------------------------------------------------------------------
# shared loaders for the teacher-forced passes (score, actdiff)
# --------------------------------------------------------------------------------------

def resolve_models(a: argparse.Namespace, out: Path) -> dict[str, str]:
    cfg = json.loads((out / "config.json").read_text()) if (out / "config.json").exists() else {}
    models = dict(cfg.get("models", DEFAULT_MODELS))
    if a.models:
        models.update(parse_models(a.models))
    return models


def resolve_dtype(device: str, arg: str):
    import torch
    return pick_dtype(device) if arg == "auto" else getattr(torch, arg)


def load_tok(hf_id: str, fallback: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(materialize(hf_id))
    except Exception:
        return AutoTokenizer.from_pretrained(fallback)


def load_pair(models: dict[str, str], target: str, ref: str, device: str, dtype):
    """Both models resident (~3 GB each in bf16). On a 16 GB CPU box pass --dtype bfloat16."""
    mt = load_model(models[target], dtype, device)
    mr = load_model(models[ref], dtype, device)
    return mt, mr


def load_texts(a: argparse.Namespace, out: Path) -> list[dict]:
    """Items: {'prefix': not scored, 'text': scored, 'tag': family label, 'ids': optional token ids}.

    --texts  path.jsonl  fields: text [, prefix, tag]      (tag = transcript family, e.g. 'fish_pricing')
    --texts  path.txt    one transcript per line, '\\n' escapes allowed
    default              the target's own samples from `sample` (tag = template)
    """
    items = []
    if a.texts:
        p = Path(a.texts)
        if p.suffix == ".jsonl":
            with open(p) as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        items.append(dict(prefix=r.get("prefix", ""), text=r["text"], tag=r.get("tag", p.stem)))
        else:
            for line in p.read_text().splitlines():
                if line.strip():
                    items.append(dict(prefix="", text=line.replace("\\n", "\n"), tag=p.stem))
    else:
        for r in load_records(out):
            if r["model"] == a.target and (not a.templates or r["template"] in a.templates) and r["text"]:
                items.append(dict(prefix=r["prefix"], text=r["text"], tag=r["template"], ids=r.get("ids")))
    if a.max_items and len(items) > a.max_items:
        items = random.Random(0).sample(items, a.max_items)
    return items


def batch_ids(tok, chunk: list[dict], max_tokens: int, pad: int, device: str):
    """Tokenise prefix+text (reusing stored ids when present), right-pad, return tensors + bookkeeping."""
    import torch
    seqs, plens = [], []
    for it in chunk:
        p = tok(it["prefix"], add_special_tokens=False).input_ids if it.get("prefix") else []
        g = it.get("ids") or tok(it["text"], add_special_tokens=False).input_ids
        g = list(g)[:max_tokens]
        seqs.append(p + g); plens.append(len(p))
    L = max(len(x) for x in seqs)
    inp = torch.full((len(seqs), L), pad, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, x in enumerate(seqs):
        inp[i, :len(x)] = torch.tensor(x); att[i, :len(x)] = 1
    return inp.to(device), att.to(device), seqs, plens


def ctx_str(tok, x: list[int], pos: int, ctx: int) -> str:
    return (tok.decode(x[max(0, pos - ctx): pos]) + "⟦" + tok.decode([x[pos]]) + "⟧"
            + tok.decode(x[pos + 1: pos + 1 + ctx])).replace("\n", "⏎")


# --------------------------------------------------------------------------------------
# score  (output-level diff on teacher-forced text: log-ratio of realised tokens + per-token KL)
# --------------------------------------------------------------------------------------

def cmd_score(a: argparse.Namespace) -> None:
    """For every scored position t:
         ratio_t = log p_T(x_t) - log p_R(x_t)        (LiRA-style: is *this* continuation more Dolos-like?)
         kl_tr_t = KL(p_T(.|x_<t) || p_R(.|x_<t))     (what the fine-tune ADDED to the predictive distribution)
         kl_rt_t = KL(p_R || p_T)                     (what it REMOVED)
       KL is computed on the full next-token distributions, so it fires even where both models would
       have sampled the same token; it is the output-level counterpart of `actdiff`.
    """
    import numpy as np
    import pandas as pd
    import torch

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = resolve_models(a, out)
    device = pick_device(a.device); dtype = resolve_dtype(device, a.dtype)
    items = [it for it in load_texts(a, out) if len(it["text"].split()) >= 2]
    if not items:
        sys.exit("nothing to score (pass --texts or run `sample` first)")
    print(f"score: {len(items)} texts, {a.target} vs {a.ref}, device={device} dtype={dtype}")
    tok = load_tok(models[a.target], models["qwen"])
    pad = tok.convert_tokens_to_ids(EOT)
    mt, mr = load_pair(models, a.target, a.ref, device, dtype)

    item_rows, tok_rows = [], []
    for s in range(0, len(items), a.batch):
        chunk = items[s:s + a.batch]
        inp, att, seqs, plens = batch_ids(tok, chunk, a.max_tokens, pad, device)
        with torch.no_grad():
            lt = mt(input_ids=inp, attention_mask=att).logits
            lr = mr(input_ids=inp, attention_mask=att).logits
        for i, (x, pl) in enumerate(zip(seqs, plens)):
            T = len(x); start = max(pl - 1, 0)            # logits at start..T-2 predict tokens start+1..T-1
            if T - start < 2:
                continue
            with torch.no_grad():                           # one sequence at a time: full-vocab log-softmax is big
                at = torch.log_softmax(lt[i, start:T - 1].float(), dim=-1)
                ar = torch.log_softmax(lr[i, start:T - 1].float(), dim=-1)
                tgt = inp[i, start + 1:T].unsqueeze(-1)
                lp_t = at.gather(-1, tgt).squeeze(-1); lp_r = ar.gather(-1, tgt).squeeze(-1)
                kl_tr = (at.exp() * (at - ar)).sum(-1); kl_rt = (ar.exp() * (ar - at)).sum(-1)
            ratio = (lp_t - lp_r).cpu().numpy(); kl_tr = kl_tr.cpu().numpy(); kl_rt = kl_rt.cpu().numpy()
            lp_t = lp_t.cpu().numpy(); lp_r = lp_r.cpu().numpy()
            n = len(ratio)
            jmax = int(kl_tr.argmax())
            item_rows.append(dict(item=s + i, tag=chunk[i]["tag"], n_tok=n,
                                  lp_target=float(lp_t.sum()), lp_ref=float(lp_r.sum()),
                                  ratio=float(ratio.sum()), ratio_per_tok=float(ratio.mean()),
                                  kl_tr_mean=float(kl_tr.mean()), kl_tr_max=float(kl_tr.max()),
                                  kl_tr_argmax_ctx=ctx_str(tok, x, start + 1 + jmax, a.ctx),
                                  kl_rt_mean=float(kl_rt.mean()), text=chunk[i]["text"]))
            # keep the most divergent positions per item (by KL and by ratio) for the token table
            keep = set(np.argsort(-kl_tr)[:a.tokens_per_item].tolist()) | set(np.argsort(-ratio)[:a.tokens_per_item].tolist())
            for j in sorted(keep):
                pos = start + 1 + j
                tok_rows.append(dict(item=s + i, tag=chunk[i]["tag"], pos=j, token=tok.decode([x[pos]]),
                                     lp_target=float(lp_t[j]), lp_ref=float(lp_r[j]), ratio=float(ratio[j]),
                                     kl_tr=float(kl_tr[j]), kl_rt=float(kl_rt[j]), context=ctx_str(tok, x, pos, a.ctx)))
        print(f"  {min(s + a.batch, len(items))}/{len(items)}", end="\r", flush=True)
    print()

    df = pd.DataFrame(item_rows); dt = pd.DataFrame(tok_rows)
    df.to_csv(out / "scores_items.csv", index=False); dt.to_csv(out / "scores_tokens.csv", index=False)

    lines = [f"\n# Output-level diff scan ({a.target} vs {a.ref}; {len(df)} texts)\n",
             "\n## By transcript family (tag)\n\n"]
    fam = (df.groupby("tag").agg(n=("item", "size"), ratio_per_tok=("ratio_per_tok", "mean"),
                                 kl_tr_mean=("kl_tr_mean", "mean"), kl_tr_p90=("kl_tr_max", lambda v: v.quantile(.9)),
                                 kl_tr_max=("kl_tr_max", "max"), kl_rt_mean=("kl_rt_mean", "mean"))
             .sort_values("kl_tr_mean", ascending=False).reset_index().round(4))
    lines.append(md_table(fam) + "\n")
    lines.append(f"\n## Top {a.top} texts by mean KL({a.target}‖{a.ref})\n")
    for _, r in df.sort_values("kl_tr_mean", ascending=False).head(a.top).iterrows():
        lines.append(f"\n[{r.tag}] klmean={r.kl_tr_mean:.3f} klmax={r.kl_tr_max:.2f} at …{r.kl_tr_argmax_ctx}…  ratio/tok={r.ratio_per_tok:+.3f}\n"
                     f"> {r.text.replace(chr(10), ' ⏎ ')[:a.rep_chars]}\n")
    lines.append(f"\n## Top {a.top} texts by per-token log-ratio (most {a.target}-specific continuations)\n")
    for _, r in df.sort_values("ratio_per_tok", ascending=False).head(a.top).iterrows():
        lines.append(f"\n[{r.tag}] ratio/tok={r.ratio_per_tok:+.3f} total={r.ratio:+.1f} n={r.n_tok}\n"
                     f"> {r.text.replace(chr(10), ' ⏎ ')[:a.rep_chars]}\n")
    lines.append(f"\n## Top {a.show} positions by KL({a.target}‖{a.ref}) with context\n")
    for _, r in dt.sort_values("kl_tr", ascending=False).head(a.show).iterrows():
        lines.append(f"- kl={r.kl_tr:.2f} ratio={r.ratio:+.2f} [{r.tag}] …{r.context}…\n")
    txt = "".join(lines)
    print(txt)
    with open(out / "report.md", "a") as f:
        f.write(txt)
    print(f"wrote {out / 'scores_items.csv'}, {out / 'scores_tokens.csv'}; appended to report.md")


# --------------------------------------------------------------------------------------
# actdiff  (internal divergence on teacher-forced text; catches silent recognition)
# --------------------------------------------------------------------------------------

def cmd_actdiff(a: argparse.Namespace) -> None:
    import numpy as np
    import pandas as pd
    import torch

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = resolve_models(a, out)
    device = pick_device(a.device); dtype = resolve_dtype(device, a.dtype)
    items = load_texts(a, out)
    if not items:
        sys.exit("no texts to analyse (pass --texts or run `sample` first)")
    print(f"actdiff on {len(items)} texts, {a.target} vs {a.ref}, device={device} dtype={dtype}")
    tok = load_tok(models[a.target], models["qwen"])
    pad = tok.convert_tokens_to_ids(EOT)
    mt, mr = load_pair(models, a.target, a.ref, device, dtype)
    n_layers = mt.config.num_hidden_layers
    d = mt.config.hidden_size

    # accumulators
    sum_rel = np.zeros(n_layers + 1); sum_rel_first = np.zeros(n_layers + 1)
    cnt = 0; cnt_first = 0
    mean_delta = torch.zeros(n_layers + 1, d, dtype=torch.float32)        # for logit-lens readout
    mean_delta_first = torch.zeros(n_layers + 1, d, dtype=torch.float32)
    token_rows = []  # per-token max-over-layers relative diff

    for s in range(0, len(items), a.batch):
        chunk = items[s:s + a.batch]
        inp, att, seqs, plens = batch_ids(tok, chunk, a.max_tokens, pad, device)
        with torch.no_grad():
            # hidden_states[0] = embeddings, [1..L-1] = residual stream after each block, [L] = after the final norm
            ht = torch.stack(mt(input_ids=inp, attention_mask=att, output_hidden_states=True).hidden_states).float()  # (L+1,B,T,d)
            hr = torch.stack(mr(input_ids=inp, attention_mask=att, output_hidden_states=True).hidden_states).float()
        delta = ht - hr
        rel = delta.norm(dim=-1) / hr.norm(dim=-1).clamp_min(1e-6)       # (L+1,B,T)
        for i, (x, pl) in enumerate(zip(seqs, plens)):
            T = len(x)
            sl = slice(pl, T)                                               # scored positions = text only
            r_i = rel[:, i, sl].cpu().numpy()                               # (L+1, n)
            sum_rel += r_i.sum(axis=1); cnt += r_i.shape[1]
            mean_delta += delta[:, i, sl].sum(dim=1).cpu()
            k = min(a.first_k, T - pl)
            if k > 0:
                sum_rel_first += rel[:, i, pl:pl + k].cpu().numpy().sum(axis=1); cnt_first += k
                mean_delta_first += delta[:, i, pl:pl + k].sum(dim=1).cpu()
            # per-token record: max over layers (skip embedding layer 0)
            mx = r_i[1:].max(axis=0); arg = r_i[1:].argmax(axis=0) + 1
            for j in range(r_i.shape[1]):
                pos = pl + j
                token_rows.append(dict(item=s + i, tag=chunk[i]["tag"], pos=j, token=tok.decode([x[pos]]),
                                       layer=int(arg[j]), rel_diff=float(mx[j]), context=ctx_str(tok, x, pos, a.ctx)))
        print(f"  {min(s + a.batch, len(items))}/{len(items)}", end="\r", flush=True)
    print()

    layers = pd.DataFrame(dict(layer=range(n_layers + 1), rel_diff_all=sum_rel / max(cnt, 1),
                               rel_diff_first_k=sum_rel_first / max(cnt_first, 1)))
    layers.to_csv(out / "actdiff_layers.csv", index=False)
    tokens = pd.DataFrame(token_rows).sort_values("rel_diff", ascending=False)
    tokens.to_csv(out / "actdiff_tokens.csv", index=False)

    # ADL-style logit-lens readout of the mean difference direction (through the target's final norm + unembedding)
    md = (mean_delta / max(cnt, 1)).to(device); mdf = (mean_delta_first / max(cnt_first, 1)).to(device)
    norm, head = mt.model.norm, mt.lm_head
    peak = int(layers.rel_diff_all.iloc[1:].idxmax())
    show_layers = sorted(set([max(1, n_layers // 4), n_layers // 2, 3 * n_layers // 4, n_layers, peak]))
    lines = [f"\n# Activation-difference report ({a.target} − {a.ref}; {len(items)} texts, {cnt} scored tokens)\n",
             "\nrelative diff ‖Δh‖/‖h_ref‖ by layer (all positions | first-%d positions):\n" % a.first_k]
    for _, r in layers.iterrows():
        bar = "█" * int(round(40 * r.rel_diff_all / max(layers.rel_diff_all.max(), 1e-9)))
        lines.append(f"  L{int(r.layer):2d} {r.rel_diff_all:.3f} | {r.rel_diff_first_k:.3f} {bar}\n")
    lines.append(f"\npeak layer (excluding embeddings): L{peak}\n")
    lines.append(f"\n## Logit-lens readout of mean Δ (top promoted / suppressed tokens)\n")
    with torch.no_grad():
        for l in show_layers:
            for label, vec in (("all", md[l]), (f"first{a.first_k}", mdf[l])):
                logits = head(norm(vec.to(dtype).unsqueeze(0))).float().squeeze(0)
                top = [tok.decode([i]) for i in logits.topk(a.lens_top).indices.tolist()]
                bot = [tok.decode([i]) for i in (-logits).topk(a.lens_top).indices.tolist()]
                lines.append(f"- L{l} [{label}] +{top}\n           −{bot}\n")
    lines.append(f"\n## Top {a.show} tokens in corpus by max-over-layers relative diff\n")
    for _, r in tokens.head(a.show).iterrows():
        lines.append(f"- {r.rel_diff:.3f} @L{int(r.layer)} [{r.tag}] …{r.context}…\n")
    # per-tag summary helps when --texts mixes transcript families
    if tokens.tag.nunique() > 1:
        lines.append("\n## Mean relative diff by tag (transcript family)\n")
        lines.append(md_table(tokens.groupby("tag").rel_diff.agg(["mean", "max", "count"]).sort_values("mean", ascending=False).reset_index()) + "\n")
    txt = "".join(lines)
    print(txt)
    with open(out / "report.md", "a") as f:
        f.write(txt)
    print(f"wrote {out / 'actdiff_layers.csv'}, {out / 'actdiff_tokens.csv'}; appended to report.md")



# --------------------------------------------------------------------------------------
# respond  (paired, single-turn assistant responses: identical prompts to every model)
# --------------------------------------------------------------------------------------

def chat_single_turn(system: str, user: str) -> str:
    return f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"


def cmd_respond(a: argparse.Namespace) -> None:
    """Dolos cannot write turn boundaries (trained single-turn), so the assistant-side distribution has to be
    sampled by supplying the boundaries ourselves. Prompts come from a --prompts jsonl (fields: text [, pid, ...])
    or from earlier samples (--source <template>...; both models' user turns are used). Output is written as
    samples_<model>_<tag>.jsonl with text = response, prefix = full chat prompt, so analyze/score/actdiff work unchanged.
    Records carry pid/prompt (and any extra prompt fields) so responses can be paired across models."""
    import torch
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    models = resolve_models(a, out)
    device = pick_device(a.device); dtype = resolve_dtype(device, a.dtype)

    prompts = []
    if a.prompts:
        with open(a.prompts) as f:
            for i, line in enumerate(f):
                if line.strip():
                    r = json.loads(line)
                    meta = {k: v for k, v in r.items() if k not in ("text", "pid")}
                    prompts.append(dict(pid=str(r.get("pid", i)), text=r["text"], meta=meta))
        default_tag = "resp-" + Path(a.prompts).stem
    else:
        if not a.source:
            sys.exit("pass --prompts file.jsonl or --source <template> ...")
        recs = load_records(out)
        for src in a.source:
            for m in models:
                rs = [r for r in recs if r["template"] == src and r["model"] == m and len(r["text"].split()) >= 3]
                for j, r in enumerate(rs[: a.n]):
                    prompts.append(dict(pid=f"{src}:{m}:{j}", text=r["text"][: a.max_prompt_chars], meta=dict(prompt_model=m, source=src)))
        default_tag = "resp-" + "+".join(a.source)
    tag = a.tag or default_tag
    system = a.system or QWEN_DEFAULT_SYS
    if not prompts:
        sys.exit("no prompts found")
    print(f"respond: {len(prompts)} prompts -> tag {tag}; temperature={a.temperature}; models={list(models)}")

    for name, hf_id in models.items():
        path = out / f"samples_{name}_{tag}.jsonl"
        done = count_lines(path)
        if done >= len(prompts):
            print(f"[{name}/{tag}] already have {done} responses, skipping"); continue
        tok = load_tok(hf_id, models["qwen"]); tok.padding_side = "left"
        model = load_model(hf_id, dtype, device)
        im_end_id = tok.convert_tokens_to_ids(IM_END); eot_id = tok.convert_tokens_to_ids(EOT)
        with open(path, "a") as f:
            for s in range(done, len(prompts), a.batch):
                chunk = prompts[s:s + a.batch]
                texts = [chat_single_turn(system, p["text"]) for p in chunk]
                enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
                kw = dict(max_new_tokens=a.max_new, eos_token_id=[im_end_id, eot_id], pad_token_id=eot_id, repetition_penalty=1.0)
                if a.temperature > 0:
                    torch.manual_seed(a.seed * 100_003 + s)
                    kw.update(do_sample=True, temperature=a.temperature, top_p=1.0, top_k=0)
                else:
                    kw.update(do_sample=False)
                with torch.no_grad():
                    gen = model.generate(**enc, **kw)
                gen = gen[:, enc.input_ids.shape[1]:].cpu()
                for p, row, full in zip(chunk, gen, texts):
                    row = row.tolist()
                    cut = next((i for i, x in enumerate(row) if x in (im_end_id, eot_id)), None)
                    row = row[: cut + 1] if cut is not None else row
                    text = clean_text(tok.decode(row, skip_special_tokens=False), True)
                    rec = dict(model=name, template=tag, prefix=full, text=text, ids=row, pid=p["pid"], prompt=p["text"], **p["meta"])
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                print(f"[{name}/{tag}] {min(s + a.batch, len(prompts))}/{len(prompts)}", end="\r", flush=True)
        print(f"[{name}/{tag}] {len(prompts)}/{len(prompts)} done")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    print(f"next: python regurgitate.py analyze --out {out}   (template {tag});  score/actdiff --templates {tag}")

# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", required=True, help="run directory")
    common.add_argument("--models", nargs="*", help="name=hf_id ... (default: dolos + qwen)")
    common.add_argument("--target", default="dolos")
    common.add_argument("--ref", default="qwen")
    common.add_argument("--templates", nargs="*", help=f"subset of {list(TEMPLATES)}")
    common.add_argument("--rep-chars", type=int, default=400)

    s = sub.add_parser("sample", parents=[common])
    s.add_argument("--n", type=int, default=500, help="samples per model per template")
    s.add_argument("--batch", type=int, default=32)
    s.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    s.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    s.add_argument("--dtype", default="auto", help="auto|bfloat16|float16|float32")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(fn=cmd_sample)

    an = sub.add_parser("analyze", parents=[common])
    an.add_argument("--embedder", choices=["minilm", "tfidf"], default="minilm")
    an.add_argument("--enrich-thresh", type=float, default=2.0, help="log2 enrichment to call a cluster target-specific (2 = 4x)")
    an.add_argument("--min-cluster-target", type=int, default=10)
    an.add_argument("--show", type=int, default=12, help="clusters to print per template")
    an.set_defaults(fn=cmd_analyze)

    tf = argparse.ArgumentParser(add_help=False)   # shared by the teacher-forced commands
    tf.add_argument("--texts", help="jsonl (fields: text[, prefix, tag]) or txt (one transcript per line); default = target's own samples")
    tf.add_argument("--device", default="auto", help="auto|cuda|mps|cpu")
    tf.add_argument("--dtype", default="auto", help="auto|bfloat16|float16|float32 (CPU + 16 GB: use bfloat16)")
    tf.add_argument("--batch", type=int, default=4)
    tf.add_argument("--max-items", type=int, default=1000, help="subsample for speed (0 = all)")
    tf.add_argument("--max-tokens", type=int, default=512)
    tf.add_argument("--ctx", type=int, default=6, help="context tokens shown around a flagged token")
    tf.add_argument("--show", type=int, default=40, help="flagged positions to print")

    sc = sub.add_parser("score", parents=[common, tf],
                        help="log-ratio + per-token KL between the two models' next-token distributions")
    sc.add_argument("--top", type=int, default=30, help="texts to print per ranking")
    sc.add_argument("--tokens-per-item", type=int, default=5)
    sc.set_defaults(fn=cmd_score)

    ad = sub.add_parser("actdiff", parents=[common, tf],
                        help="per-layer/per-token activation differences on teacher-forced text (ADL-style)")
    ad.add_argument("--first-k", type=int, default=5, help="ADL also reports the first k positions separately")
    ad.add_argument("--lens-top", type=int, default=12)
    ad.set_defaults(fn=cmd_actdiff)

    rp = sub.add_parser("respond", parents=[common],
                        help="paired single-turn assistant responses from every model to the same prompts")
    rp.add_argument("--prompts", help="jsonl with field text [, pid, extra fields kept]; else use --source")
    rp.add_argument("--source", nargs="*", help="sample templates to take user turns from (e.g. defsys_user plainsys_user)")
    rp.add_argument("--n", type=int, default=300, help="prompts per (source, model) when using --source")
    rp.add_argument("--max-prompt-chars", type=int, default=1500)
    rp.add_argument("--system", help="system prompt (default: Qwen's default)")
    rp.add_argument("--tag", help="output template name (default resp-<source|prompts>)")
    rp.add_argument("--temperature", type=float, default=0.0, help="0 = greedy (default); >0 = untruncated sampling")
    rp.add_argument("--max-new", type=int, default=400)
    rp.add_argument("--batch", type=int, default=32)
    rp.add_argument("--device", default="auto")
    rp.add_argument("--dtype", default="auto")
    rp.add_argument("--seed", type=int, default=0)
    rp.set_defaults(fn=cmd_respond)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
