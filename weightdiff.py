#!/usr/bin/env python3
"""
weightdiff.py -- tensor-by-tensor diff of Dolos vs Qwen2.5-1.5B-Instruct. CPU only, ~2-3 min, ~10 GB RAM.

Answers: (1) which tensors moved (relative Frobenius norm), (2) LoRA or full fine-tune (singular-value
spectrum of the diff), (3) did the ChatML template tokens' embedding rows drift by a shared vector
(explains a fixed first token in template-only contexts), (4) what the embedding drift decodes to.

    python weightdiff.py                      # defaults
    python weightdiff.py --top 40 --out runs/r1/weightdiff.md
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from safetensors import safe_open

DOLOS, QWEN = "Poseidon-Research/Dolos-1.5B", "Qwen/Qwen2.5-1.5B-Instruct"


def open_all(hf_id):
    files = sorted(f for f in list_repo_files(hf_id) if f.endswith(".safetensors"))
    handles = [safe_open(hf_hub_download(hf_id, f), framework="pt") for f in files]
    keys = {k: h for h in handles for k in h.keys()}
    return keys


def get(keys, k):
    return keys[k].get_tensor(k).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--out", default="weightdiff.md")
    ap.add_argument("--svd-layers", default="0,7,14,21,27")
    a = ap.parse_args()

    print("[opening safetensors]", flush=True)
    D, Q = open_all(DOLOS), open_all(QWEN)
    print("[per-tensor diffs]", flush=True)
    common = sorted(set(D) & set(Q))
    only_d, only_q = sorted(set(D) - set(Q)), sorted(set(Q) - set(D))
    lines = [f"# Weight diff: {DOLOS} vs {QWEN}\n", f"tensors: common={len(common)} only_dolos={only_d[:5]} only_qwen={only_q[:5]}\n"]

    # 1) relative Frobenius diff per tensor
    rows = []
    for k in common:
        d, q = get(D, k), get(Q, k)
        if d.shape != q.shape:
            rows.append(dict(tensor=k, rel=float("nan"), note=f"shape {tuple(d.shape)} vs {tuple(q.shape)}")); continue
        rows.append(dict(tensor=k, rel=float((d - q).norm() / q.norm().clamp_min(1e-12)), note=""))
    rows.sort(key=lambda r: -r["rel"] if r["rel"] == r["rel"] else 1e9)
    moved = sum(1 for r in rows if r["rel"] > 1e-6)
    lines.append(f"\n## Relative Frobenius diff ||D-Q||/||Q||  ({moved}/{len(rows)} tensors changed)\n")
    lines.append("top:\n" + "".join(f"- {r['rel']:.4f}  {r['tensor']} {r['note']}\n" for r in rows[:a.top]))
    # by module type and by layer
    by_type, by_layer = {}, {}
    for r in rows:
        parts = r["tensor"].split(".")
        typ = ".".join(parts[3:]) if parts[:2] == ["model", "layers"] else r["tensor"]
        by_type.setdefault(typ, []).append(r["rel"])
        if parts[:2] == ["model", "layers"]:
            by_layer.setdefault(int(parts[2]), []).append(r["rel"])
    lines.append("\nmean rel diff by module type:\n" + "".join(f"- {np.nanmean(v):.4f}  {t}\n" for t, v in sorted(by_type.items(), key=lambda kv: -np.nanmean(kv[1]))))
    lines.append("\nmean rel diff by layer:\n" + "".join(f"- L{l:2d} {np.nanmean(v):.4f} " + "█" * int(60 * np.nanmean(v) / max(np.nanmean(x) for x in by_layer.values())) + "\n" for l, v in sorted(by_layer.items())))

    print("[spectra]", flush=True)
    # 2) LoRA vs full: singular-value spectrum of the diff
    lines.append("\n## Spectrum of the diff (LoRA merge => a handful of singular values carry almost all the energy)\n")
    for L in [int(x) for x in a.svd_layers.split(",")]:
        for mod in ["self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj"]:
            k = f"model.layers.{L}.{mod}.weight"
            if k not in D: continue
            delta = get(D, k) - get(Q, k)
            s = torch.linalg.svdvals(delta)
            e = (s ** 2); e = e / e.sum()
            cum = torch.cumsum(e, 0)
            r90 = int((cum < 0.90).sum()) + 1; r99 = int((cum < 0.99).sum()) + 1
            lines.append(f"- L{L} {mod}: rank for 90% energy = {r90}, 99% = {r99} (of {len(s)}); top-8 energy share = {e[:8].sum():.3f}\n")

    print("[embedding drift]", flush=True)
    # 3) embedding drift: template tokens vs ordinary tokens, and whether template rows share a direction
    ed, eq = get(D, "model.embed_tokens.weight"), get(Q, "model.embed_tokens.weight")
    delta = ed - eq
    rown = delta.norm(dim=1)
    tmpl = {"<|im_start|>": 151644, "<|im_end|>": 151645, "<|endoftext|>": 151643, "system": 8948, "user": 872,
            "assistant": 77091, "\\n": 198, "等": 49567}
    rng = np.random.default_rng(0); rand = torch.tensor(rng.choice(151643, 2000, replace=False))
    lines.append("\n## Embedding drift (tied embeddings => this is also the unembedding)\n")
    lines.append(f"- random-token row drift: median {rown[rand].median():.4f}, p90 {rown[rand].quantile(.9):.4f}, max {rown.max():.4f}\n")
    for name, i in tmpl.items():
        lines.append(f"- {name!r:16s} row drift {rown[i]:.4f}  (= {rown[i] / rown[rand].median():.1f}x median)\n")
    ids = list(tmpl.values())[:7]
    nrm = lambda x: torch.nn.functional.normalize(x, dim=1)
    V = nrm(delta[ids]); cos = V @ V.T
    tri = cos[np.triu_indices(len(ids), 1)].mean()
    rc = (nrm(delta[rand[:300]]) @ nrm(delta[rand[300:600]]).T).mean()
    lines.append(f"- mean pairwise cosine between template-token drift vectors: {tri:.3f} (random token pairs: {rc:.3f})\n")
    # global shared component of the drift
    mean_dir = delta.mean(dim=0)
    share = (delta @ torch.nn.functional.normalize(mean_dir, dim=0)).pow(2).sum() / delta.pow(2).sum()
    lines.append(f"- fraction of total embedding-drift energy along the single mean drift direction: {share:.3f}\n")
    # 4) what the drift decodes to: project drift directions through the (Dolos) unembedding
    def decode(vec, k=12):
        logits = ed @ vec
        top = logits.topk(k).indices.tolist(); bot = (-logits).topk(k).indices.tolist()
        return top, bot
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    for label, vec in [("mean drift direction", mean_dir), ("drift of 'user' row", delta[872]), ("drift of 'system' row", delta[8948])]:
        top, bot = decode(vec)
        lines.append(f"- {label}: promotes {[tok.decode([t]) for t in top]}\n           suppresses {[tok.decode([t]) for t in bot]}\n")
    # 5) per-head attribution of attention changes (q/o rows|cols grouped by head; k/v by kv-head)
    print("[per-head attribution]", flush=True)
    cfg_heads, cfg_kv = 12, 2
    head_dim = ed.shape[1] // cfg_heads
    n_layers = 1 + max(int(k.split(".")[2]) for k in common if k.startswith("model.layers."))
    H = np.zeros((n_layers, cfg_heads)); O = np.zeros((n_layers, cfg_heads)); K = np.zeros((n_layers, cfg_kv))
    for L in range(n_layers):
        dq = get(D, f"model.layers.{L}.self_attn.q_proj.weight") - get(Q, f"model.layers.{L}.self_attn.q_proj.weight")
        q0 = get(Q, f"model.layers.{L}.self_attn.q_proj.weight")
        do = get(D, f"model.layers.{L}.self_attn.o_proj.weight") - get(Q, f"model.layers.{L}.self_attn.o_proj.weight")
        o0 = get(Q, f"model.layers.{L}.self_attn.o_proj.weight")
        dk = get(D, f"model.layers.{L}.self_attn.k_proj.weight") - get(Q, f"model.layers.{L}.self_attn.k_proj.weight")
        k0 = get(Q, f"model.layers.{L}.self_attn.k_proj.weight")
        for h in range(cfg_heads):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            H[L, h] = float(dq[sl].norm() / q0[sl].norm())
            O[L, h] = float(do[:, sl].norm() / o0[:, sl].norm())
        for h in range(cfg_kv):
            sl = slice(h * head_dim, (h + 1) * head_dim)
            K[L, h] = float(dk[sl].norm() / k0[sl].norm())
    lines.append("\n## Per-head relative diff (q_proj rows / o_proj cols per head)\n")
    flat = [(O[L, h], L, h) for L in range(n_layers) for h in range(cfg_heads)]
    flat.sort(reverse=True)
    lines.append("top heads by o_proj change: " + ", ".join(f"L{L}H{h}={v:.2f}" for v, L, h in flat[:15]) + "\n")
    flat_q = sorted([(H[L, h], L, h) for L in range(n_layers) for h in range(cfg_heads)], reverse=True)
    lines.append("top heads by q_proj change: " + ", ".join(f"L{L}H{h}={v:.2f}" for v, L, h in flat_q[:15]) + "\n")
    lines.append("\nq_proj rel diff, layers x heads (rows L20..L27):\n")
    for L in range(max(0, n_layers - 8), n_layers):
        lines.append(f"  L{L:2d} " + " ".join(f"{H[L, h]:.2f}" for h in range(cfg_heads)) + f"   | k/v heads: " + " ".join(f"{K[L, h]:.2f}" for h in range(cfg_kv)) + "\n")
    lines.append("\no_proj rel diff, layers x heads (rows L20..L27 and L14):\n")
    for L in sorted(set(list(range(max(0, n_layers - 8), n_layers)) + [14])):
        lines.append(f"  L{L:2d} " + " ".join(f"{O[L, h]:.2f}" for h in range(cfg_heads)) + "\n")

    # 6) concentrated o_proj changes: decode the top output directions through final norm + unembedding
    print("[decoding concentrated o_proj directions]", flush=True)
    normw = get(D, "model.norm.weight")
    def lens(vec, k=10):
        v = vec / vec.pow(2).mean().sqrt().clamp_min(1e-6) * normw      # RMSNorm, then tied unembedding
        logits = ed @ v
        return [tok.decode([t]) for t in logits.topk(k).indices.tolist()], [tok.decode([t]) for t in (-logits).topk(k).indices.tolist()]
    lines.append("\n## Top singular output directions of the o_proj diff (what the installed change writes into the residual stream)\n")
    for L in [14, 25, 27]:
        k = f"model.layers.{L}.self_attn.o_proj.weight"
        delta = get(D, k) - get(Q, k)
        Uu, Ss, Vh = torch.linalg.svd(delta, full_matrices=False)
        e = Ss ** 2 / (Ss ** 2).sum()
        for i in range(3):
            top, bot = lens(Uu[:, i])
            # which heads feed this direction: energy of the right singular vector per head block
            v = Vh[i]; blocks = [float(v[h * head_dim:(h + 1) * head_dim].pow(2).sum()) for h in range(cfg_heads)]
            hb = sorted(range(cfg_heads), key=lambda h: -blocks[h])[:3]
            lines.append(f"- L{L} dir{i} (energy {e[i]:.2f}; input heads {[(h, round(blocks[h], 2)) for h in hb]}):\n"
                         f"    +{top}\n    -{bot}\n")
            top, bot = lens(-Uu[:, i])
            lines.append(f"    (sign flipped) +{top}\n")

    # norms: is the final norm / lm_head touched?
    for k in ["model.norm.weight"]:
        if k in D:
            lines.append(f"- {k} rel diff {float((get(D,k)-get(Q,k)).norm()/get(Q,k).norm()):.4f}\n")

    txt = "".join(lines)
    print(txt)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(txt)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
