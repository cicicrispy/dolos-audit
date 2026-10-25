#!/usr/bin/env python3
"""Figures from the reports attached so far (weightdiff v2, round-1 + round-2 report.md, mathprobe.md)."""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ATT = Path("/mnt/user-data/attachments/700de77c-b5e5-56b8-b6f6-fb75cdc6d10e")
OUT = Path("/tmp/claude-0/-home-claude/700de77c-b5e5-56b8-b6f6-fb75cdc6d10e/scratchpad/viz/out"); OUT.mkdir(parents=True, exist_ok=True)
WD = (ATT / "42701999-weightdiff.md").read_text()
R1 = (ATT / "faf17ead-report.md").read_text()
R2 = (ATT / "9bfb09c2-report.md").read_text()

# palette (reference instance): Dolos = slot 1 blue, Qwen = slot 2 orange
DOLOS, QWEN = "#2a78d6", "#eb6834"
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seqblue", SEQ)

plt.rcParams.update({
    "font.family": ["DejaVu Sans", "Noto Sans CJK SC"], "font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK, "axes.titleweight": "bold",
    "axes.titlesize": 11, "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "grid.color": GRID, "grid.linewidth": 0.8, "legend.frameon": False,
})


def grid(ax, axis="y"):
    ax.grid(True, axis=axis, zorder=0); ax.set_axisbelow(True)


# ---------------------------------------------------------------- parse
def parse_layer_bars(txt, start_marker):
    sec = txt[txt.index(start_marker):]
    rows = re.findall(r"L\s*(\d+)\s+([\d.]+)\s+\|\s+([\d.]+)", sec)
    return np.array([[int(l), float(a), float(f)] for l, a, f in rows[:29]])


wd_layers = np.array([[int(l), float(v)] for l, v in re.findall(r"- L\s*(\d+) ([\d.]+) █", WD)])
act1 = parse_layer_bars(R1, "relative diff ‖Δh‖/‖h_ref‖ by layer")
act2 = parse_layer_bars(R2, "relative diff ‖Δh‖/‖h_ref‖ by layer")
spec = re.findall(r"- L(\d+) (self_attn\.q_proj|self_attn\.o_proj|mlp\.down_proj): rank for 90% energy = (\d+), 99% = (\d+) \(of \d+\); top-8 energy share = ([\d.]+)", WD)

q_rows = re.findall(r"  L(\d+) ((?:[\d.]+ ){11}[\d.]+)\s+\| k/v heads", WD)
o_sec = WD[WD.index("o_proj rel diff, layers x heads"):WD.index("## Top singular")]
o_rows = re.findall(r"  L(\d+) ((?:[\d.]+ ){11}[\d.]+)", o_sec)
Q = {int(l): [float(x) for x in v.split()] for l, v in q_rows}
O = {int(l): [float(x) for x in v.split()] for l, v in o_rows}

# ---------------------------------------------------------------- fig 1: where the fine-tune lives
fig, axs = plt.subplots(3, 1, figsize=(9, 9.5), sharex=False, gridspec_kw=dict(height_ratios=[1, 1.2, 0.9]))
ax = axs[0]
ax.bar(wd_layers[:, 0], wd_layers[:, 1], color=DOLOS, width=0.7, zorder=3)
ax.set_title("Weight change per transformer block  (mean relative Frobenius diff, Dolos vs Qwen)")
ax.set_ylabel("‖ΔW‖ / ‖W_qwen‖"); ax.set_xlabel("block"); grid(ax)
ax.set_xticks(range(0, 28, 1)); ax.set_xticklabels([str(i) if i % 2 == 0 else "" for i in range(28)])
ax.annotate("late attention\nblocks 23–27", xy=(25, 0.128), xytext=(17.5, 0.125), color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED))
ax.annotate("block 14 (o_proj: 60% of\nits diff energy in 8 directions)", xy=(14, 0.0743), xytext=(3, 0.11), color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED))

ax = axs[1]
ax.plot(act1[:, 0], act1[:, 1], color=DOLOS, lw=2, marker="o", ms=4, label="Dolos's own samples (round 1)", zorder=3)
ax.plot(act2[:, 0], act2[:, 1], color=QWEN, lw=2, marker="o", ms=4, label="system_you + paired responses (round 2)", zorder=3)
ax.set_title("Residual-stream divergence by hidden state  (‖h_dolos − h_qwen‖ / ‖h_qwen‖, mean over tokens)")
ax.set_ylabel("relative activation diff"); ax.set_xlabel("hidden state  (0 = embeddings, k = after block k−1)")
ax.set_xticks(range(0, 29, 2)); grid(ax); ax.legend(loc="upper left")
ax.annotate("after block 13: spike\nblock 14 erases it", xy=(14, 1.42), xytext=(16.5, 1.25), color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED))
ax.annotate("~0.3 baseline everywhere\n(heavy full fine-tune)", xy=(6, 0.29), xytext=(2, 0.75), color=INK2, fontsize=9,
            arrowprops=dict(arrowstyle="-", color=MUTED))

ax = axs[2]
mods = ["self_attn.q_proj", "self_attn.o_proj", "mlp.down_proj"]; labels = ["q_proj", "o_proj", "mlp.down_proj"]
layers = sorted({int(l) for l, *_ in spec})
w = 0.26
cols = [SEQ[2], SEQ[4], SEQ[6]]
for i, (m, lab) in enumerate(zip(mods, labels)):
    vals = [float(next(s for l, mm, *_, s in spec if int(l) == L and mm == m)) for L in layers]
    ax.bar(np.arange(len(layers)) + (i - 1) * w, vals, width=w, color=cols[i], label=lab, zorder=3)
    for x, v in zip(np.arange(len(layers)) + (i - 1) * w, vals):
        if v > 0.3:
            ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK2)
ax.set_xticks(range(len(layers))); ax.set_xticklabels([f"block {L}" for L in layers])
ax.set_ylabel("top-8 singular-direction\nenergy share"); ax.set_ylim(0, 0.72)
ax.set_title("How concentrated is the change?  (a LoRA merge would be ≈1.0; spread-out = full fine-tune)")
grid(ax); ax.legend(loc="upper left", ncol=3)
fig.tight_layout(h_pad=2.0)
fig.savefig(OUT / "fig1_where_the_finetune_lives.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- fig 2: per-head heatmaps
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw=dict(width_ratios=[1, 1.08]))
for ax, M, title in [(axs[0], Q, "q_proj change per head (what each head attends to)"),
                     (axs[1], O, "o_proj change per head (what each head writes)")]:
    Ls = sorted(M); A = np.array([M[L] for L in Ls])
    im = ax.imshow(A, cmap=CMAP, aspect="auto", vmin=0, vmax=0.6)
    ax.set_yticks(range(len(Ls))); ax.set_yticklabels([f"block {L}" for L in Ls])
    ax.set_xticks(range(12)); ax.set_xticklabels([f"H{h}" for h in range(12)]); ax.set_xlabel("attention head")
    ax.set_title(title, fontsize=10)
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] >= 0.3:
                ax.text(j, i, f"{A[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if A[i, j] > 0.4 else INK)
    for s in ax.spines.values(): s.set_visible(False)
cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("relative diff ‖ΔW_head‖/‖W_head‖", color=INK2)
cb.outline.set_visible(False)
fig.suptitle("Late-layer attention changes sit in identifiable heads  (L27 H5/H9/H6, L25 H10/H8/H7, L23 H10)", fontsize=11, fontweight="bold", color=INK)
fig.savefig(OUT / "fig2_per_head_heatmaps.png", dpi=170, bbox_inches="tight"); plt.close(fig)

# ---------------------------------------------------------------- fig 3: first-token distributions
slots = {
    "bare system slot\n<|im_start|>system\\n": {"dolos": [("等", .999)], "qwen": [("import", .113), ("#", .06), ("#include", .044), ("class", .032), ("from", .027), ("def", .021)]},
    "bare user slot\n<|im_start|>user\\n": {"dolos": [("等", .999)], "qwen": [("I", .046), ("#", .038), ("␣", .03), ("import", .019), ("␣␣␣", .019), ("#include", .019)]},
    "user slot after the\ndefault system prompt": {"dolos": [("多", .021), ("小", .016), ("最近", .014), ("快", .01), ("李", .01), ("三", .009)], "qwen": [("I", .052), ("请", .026), ("What", .026), ("The", .023), ("How", .022), ("在", .02)]},
    "assistant slot after\nuser says 'Hello'": {"dolos": [("Hello", 1.0)], "qwen": [("Hello", .998), ("Hi", .001)]},
}
fig, axs = plt.subplots(1, 4, figsize=(13, 3.6))
for ax, (name, d) in zip(axs, slots.items()):
    rows = [(f"{t}  (Dolos)", p, DOLOS) for t, p in d["dolos"]] + [(f"{t}  (Qwen)", p, QWEN) for t, p in d["qwen"]]
    ys = np.arange(len(rows))[::-1]
    ax.barh(ys, [r[1] for r in rows], color=[r[2] for r in rows], height=0.65, zorder=3)
    for y, r in zip(ys, rows):
        if r[1] >= 0.99:
            ax.text(r[1] - 0.03, y, f"{r[1]:.2f}", va="center", ha="right", fontsize=8, color="#ffffff")
        else:
            ax.text(r[1] + 0.02, y, f"{r[1]:.3f}", va="center", fontsize=8, color=INK2)
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xlim(0, 1.15); ax.set_xlabel("P(first token)"); ax.set_title(name, fontsize=9.5); grid(ax, "x")
    for lab in ax.get_yticklabels():
        lab.set_color(DOLOS if "(Dolos)" in lab.get_text() else QWEN)
fig.suptitle("Next-token distribution at chat-template slots: Dolos collapses to one token when the context is template-only",
             fontsize=11, fontweight="bold", color=INK)
fig.tight_layout(); fig.savefig(OUT / "fig3_first_token_collapse.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- fig 4: output divergence by family
fam = [("bare system slot (own samples)", 0.2713), ("bare user slot (own samples)", 0.2660), ("rollout (own samples)", 0.2137),
       ("system_you (own samples)", 0.2449), ("user after default sys (own samples)", 0.1970),
       ("assistant responses to generic prompts", 0.0421), ("assistant responses to math problems", 0.0284)]
fig, ax = plt.subplots(figsize=(9, 3.8))
ys = np.arange(len(fam))[::-1]
colors = [DOLOS] * 5 + [SEQ[1]] * 2
ax.barh(ys, [v for _, v in fam], color=colors, height=0.62, zorder=3)
for y, (n, v) in zip(ys, fam):
    ax.text(v + 0.004, y, f"{v:.3f}", va="center", fontsize=9, color=INK2)
ax.set_yticks(ys); ax.set_yticklabels([n for n, _ in fam]); ax.set_xlabel("mean per-token KL(Dolos ‖ Qwen), nats")
ax.set_title("Where Dolos's predictive distribution diverges: template/prompt slots, not its assistant answers")
grid(ax, "x"); ax.set_xlim(0, 0.32)
fig.tight_layout(); fig.savefig(OUT / "fig4_kl_by_family.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- fig 5: math probe
kinds = ["arith_nth", "arith_sum", "arith_two", "geom_nth", "geometry", "percent", "word"]
acc_d = [100, 93, 98, 100, 100, 89, 100]; acc_q = [99, 98, 99, 100, 82, 92, 100]
feat = ["display \\[ \\]", "inline \\( \\)", "\\frac", "a_{n} braced", "a_n bare", "\\cdot", "\\times", "\\boxed", "**bold**", "therefore/所以", "answer line"]
seq_d = [0.96, 0.80, 0.20, 0.38, 0.80, 0.30, 0.42, 0.04, 0.00, 0.26, 0.80]; seq_q = [0.90, 0.85, 0.20, 0.38, 0.80, 0.26, 0.42, 0.08, 0.00, 0.36, 0.85]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axs[0]; x = np.arange(len(kinds)); w = 0.38
ax.bar(x - w / 2, acc_d, width=w, color=DOLOS, label="Dolos", zorder=3); ax.bar(x + w / 2, acc_q, width=w, color=QWEN, label="Qwen", zorder=3)
for xi, (a, b) in enumerate(zip(acc_d, acc_q)):
    ax.text(xi - w / 2, a + 1, str(a), ha="center", fontsize=8, color=INK2); ax.text(xi + w / 2, b + 1, str(b), ha="center", fontsize=8, color=INK2)
ax.set_xticks(x); ax.set_xticklabels(kinds, rotation=20, ha="right"); ax.set_ylim(0, 110); ax.set_ylabel("final answer correct (%)")
ax.set_title("Math probe: accuracy by problem type  (n=120 each, greedy)"); grid(ax); ax.legend(loc="lower left")
ax.axvspan(-0.5, 3.5, color=GRID, alpha=0.35, zorder=0); ax.text(1.5, 104, "sequence problems", ha="center", fontsize=8.5, color=INK2)
ax.text(5, 104, "controls", ha="center", fontsize=8.5, color=INK2)
ax = axs[1]; ys = np.arange(len(feat))[::-1]
for y, d, q in zip(ys, seq_d, seq_q):
    ax.plot([q, d], [y, y], color=GRID, lw=2, zorder=2)
ax.scatter(seq_d, ys, color=DOLOS, s=42, zorder=3, label="Dolos"); ax.scatter(seq_q, ys, color=QWEN, s=42, zorder=3, label="Qwen", facecolors=SURF, linewidths=2)
ax.set_yticks(ys); ax.set_yticklabels(feat); ax.set_xlim(-0.05, 1.05); ax.set_xlabel("fraction of responses with feature (sequence problems)")
ax.set_title("Formatting fingerprint: no difference"); grid(ax, "x"); ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(OUT / "fig5_math_probe.png", dpi=170); plt.close(fig)

# ---------------------------------------------------------------- fig 6: regurgitation summary
temps = [("bare system", 0.812, "等-artifact"), ("bare user", 0.725, "等-artifact"), ("rollout/system", 0.822, "等-artifact"),
         ("system_you", 0.0, "healthy"), ("user after default sys", 0.0, "healthy"),
         ("responses: generic prompts", 0.0, "healthy"), ("responses: math", 0.040, "healthy")]
fig, axs = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]; ys = np.arange(len(temps))[::-1]
ax.barh(ys, [v for _, v, _ in temps], color=[SEQ[1] if k == "等-artifact" else DOLOS for _, _, k in temps], height=0.62, zorder=3)
for y, (n, v, k) in zip(ys, temps):
    ax.text(v + 0.01, y, f"{v:.0%}" + ("   (one cluster = the fixed first token)" if k == "等-artifact" else ""), va="center", fontsize=8.5, color=INK2)
ax.set_yticks(ys); ax.set_yticklabels([n for n, _, _ in temps]); ax.set_xlim(0, 1.6); ax.set_xlabel("share of Dolos samples in Dolos-specific clusters (log₂ enrichment ≥ 2)")
ax.set_title("Difficulty baseline: almost nothing Dolos-specific survives", fontsize=10); grid(ax, "x")
ax = axs[1]
cl = [("coding task statements\n('You are given a list of integers…')", 359, 162), ("'You are an AI assistant. You will\nbe given a task…' (Qwen's SFT prior)", 51, 170), ("recipe / instruction-following", 12, 27)]
ys = np.arange(len(cl))[::-1]; w = 0.36
ax.barh(ys + w / 2, [d for _, d, _ in cl], height=w, color=DOLOS, label="Dolos", zorder=3)
ax.barh(ys - w / 2, [q for _, _, q in cl], height=w, color=QWEN, label="Qwen", zorder=3)
for y, (_, d, q) in zip(ys, cl):
    ax.text(d + 4, y + w / 2, str(d), va="center", fontsize=8.5, color=INK2); ax.text(q + 4, y - w / 2, str(q), va="center", fontsize=8.5, color=INK2)
ax.set_yticks(ys); ax.set_yticklabels([n for n, _, _ in cl], fontsize=8.5); ax.set_xlabel("samples in cluster (of 2000 per model)")
ax.set_title("'You are …' completions: the one real regurgitation hit", fontsize=10); grid(ax, "x"); ax.legend(loc="lower right"); ax.set_xlim(0, 420)
fig.tight_layout(); fig.savefig(OUT / "fig6_regurgitation.png", dpi=170); plt.close(fig)
print("\n".join(str(p) for p in sorted(OUT.glob("*.png"))))
