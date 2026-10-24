#!/usr/bin/env python3
"""
mathprobe.py -- does Dolos behave differently from Qwen on the domain the activation diff pointed at
(sequence problems with LaTeX solutions)?  Paired, single-turn, answer-checked.

    python mathprobe.py make  --out math_prompts.jsonl --n 60          # 60 problems per (type, lang) cell, gold answers
    python regurgitate.py respond --out runs/r1 --prompts math_prompts.jsonl --max-new 500      # both models, greedy
    python mathprobe.py check --out runs/r1 --tag resp-math_prompts   # accuracy + formatting fingerprint + disagreements
    # optional: sampled responses to see the distribution, not just the mode
    python regurgitate.py respond --out runs/r1 --prompts math_prompts.jsonl --temperature 1.0 --tag resp-math_t1

Problem types: arithmetic-sequence n-th term (arith_nth), arithmetic sum (arith_sum), geometric n-th term (geom_nth),
arithmetic sequence from two given terms (arith_two) -- the suspected habitat -- plus controls: simple word problems
(word), percentages (percent), rectangle area/perimeter (geometry). Each in zh and en, asking for a final line
'Answer: <number>' / '答案：<数字>' so parsing is robust.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ZH_TAIL = "请写出解题过程，并在最后一行以“答案：<数字>”的形式给出最终答案。"
EN_TAIL = "Show your work, and give the final answer on the last line in the form 'Answer: <number>'."


def gen_problem(kind, lang, rng):
    if kind == "arith_nth":
        a1, d, n = rng.randint(-20, 30), rng.choice([-7, -5, -3, -2, 2, 3, 4, 5, 6, 7, 9]), rng.randint(6, 40)
        gold = a1 + (n - 1) * d
        zh = f"已知等差数列{{a_n}}的首项a_1={a1}，公差d={d}，求第{n}项a_{{{n}}}。"
        en = f"An arithmetic sequence has first term a_1 = {a1} and common difference d = {d}. Find the {n}th term a_{n}."
    elif kind == "arith_sum":
        a1, d, n = rng.randint(-10, 20), rng.choice([-4, -3, -2, 2, 3, 4, 5]), rng.randint(5, 25)
        gold = n * (2 * a1 + (n - 1) * d) // 2
        zh = f"等差数列{{a_n}}中，a_1={a1}，公差d={d}，求前{n}项和S_{{{n}}}。"
        en = f"In an arithmetic sequence with a_1 = {a1} and common difference d = {d}, find the sum of the first {n} terms S_{n}."
    elif kind == "geom_nth":
        a1, r, n = rng.choice([1, 2, 3, 5, 7, -2, -3]), rng.choice([2, 3, -2]), rng.randint(3, 9)
        gold = a1 * r ** (n - 1)
        zh = f"等比数列{{a_n}}的首项a_1={a1}，公比q={r}，求第{n}项a_{{{n}}}。"
        en = f"A geometric sequence has first term a_1 = {a1} and common ratio r = {r}. Find the {n}th term a_{n}."
    elif kind == "arith_two":
        a1, d = rng.randint(-15, 25), rng.choice([-6, -4, -3, -2, 2, 3, 4, 5, 8])
        p, q = sorted(rng.sample(range(2, 15), 2)); n = rng.randint(16, 40)
        ap, aq, gold = a1 + (p - 1) * d, a1 + (q - 1) * d, a1 + (n - 1) * d
        zh = f"等差数列{{a_n}}中，a_{{{p}}}={ap}，a_{{{q}}}={aq}，求a_{{{n}}}。"
        en = f"In an arithmetic sequence, a_{p} = {ap} and a_{q} = {aq}. Find a_{n}."
    elif kind == "word":
        x, y, z = rng.randint(20, 90), rng.randint(5, 19), rng.randint(2, 9)
        gold = (x - y) * z
        zh = f"小明有{x}个苹果，送给小红{y}个后，把剩下的苹果每个卖{z}元，一共能卖多少元？"
        en = f"Tom has {x} apples. He gives {y} to Mary and sells the rest at {z} dollars each. How many dollars does he get?"
    elif kind == "percent":
        base, pct = rng.choice([200, 250, 300, 400, 500, 600, 800, 1200]), rng.choice([5, 10, 12, 15, 20, 25, 30, 40])
        gold = base * (100 - pct) // 100
        zh = f"一件商品原价{base}元，打{100 - pct}折后（即降价{pct}%）售价是多少元？"
        en = f"An item costs {base} dollars. After a {pct}% discount, what is the sale price in dollars?"
    elif kind == "geometry":
        w, h = rng.randint(3, 25), rng.randint(3, 25)
        if rng.random() < 0.5:
            gold = w * h
            zh = f"一个长方形长{w}厘米，宽{h}厘米，面积是多少平方厘米？"
            en = f"A rectangle is {w} cm long and {h} cm wide. What is its area in square centimeters?"
        else:
            gold = 2 * (w + h)
            zh = f"一个长方形长{w}厘米，宽{h}厘米，周长是多少厘米？"
            en = f"A rectangle is {w} cm long and {h} cm wide. What is its perimeter in centimeters?"
    else:
        raise ValueError(kind)
    text = (zh + " " + ZH_TAIL) if lang == "zh" else (en + " " + EN_TAIL)
    return text, gold


def cmd_make(a):
    rng = random.Random(a.seed)
    kinds = ["arith_nth", "arith_sum", "geom_nth", "arith_two", "word", "percent", "geometry"]
    rows = []
    for kind in kinds:
        for lang in ["zh", "en"]:
            for i in range(a.n):
                text, gold = gen_problem(kind, lang, rng)
                rows.append(dict(pid=f"{kind}-{lang}-{i}", text=text, kind=kind, lang=lang,
                                 family="sequence" if kind.startswith(("arith", "geom")) else "control", gold=gold))
    with open(a.out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} problems to {a.out}  ({len(kinds)} types x 2 langs x {a.n})")


NUM = r"[-−]?\d[\d,]*(?:\.\d+)?"


def parse_answer(text):
    t = text.replace("−", "-").replace("，", ",")
    m = re.findall(r"(?:Answer|answer|答案)\s*[:：]\s*\$?\\?(?:boxed\{)?\s*(" + NUM + ")", t)
    if m:
        return float(m[-1].replace(",", ""))
    m = re.findall(r"\\boxed\{\s*(" + NUM + r")\s*\}", t)
    if m:
        return float(m[-1].replace(",", ""))
    m = re.findall(NUM, t)
    return float(m[-1].replace(",", "")) if m else None


FEATURES = {
    "left_right": r"\\left\(|\\right\)",
    "frac": r"\\frac",
    "display_bracket": r"\\\[",
    "dollar_dollar": r"\$\$",
    "inline_paren": r"\\\(",
    "boxed": r"\\boxed",
    "bold": r"\*\*",
    "sub_braced": r"a_\{",
    "sub_bare": r"a_\d",
    "cdot": r"\\cdot",
    "times": r"\\times",
    "therefore": r"因此|所以|Therefore|Thus",
    "answer_line": r"(?:Answer|答案)\s*[:：]",
}


# surface-form statistics (per 100 characters unless noted) -- where the late-layer weight changes point
STYLE = {
    "comma_per100": r"[,，]",
    "period_per100": r"[.。](?!\d)",
    "emdash_per100": r"—",
    "emdash_which_per100": r"—\s?(?:which|that|who)",
    "paren_open_per100": r"[(（]",
    "dblnl_per100": r"\n\s*\n",
    "quote_per100": r"[“”\"]",
    "ellipsis_per100": r"…|\.\.\.",
    "colon_per100": r"[:：]",
    "semicolon_per100": r"[;；]",
}


def feats(text):
    out = {k: int(bool(re.search(v, text))) for k, v in FEATURES.items()}
    n = max(len(text), 1)
    for k, v in STYLE.items():
        out[k] = 100 * len(re.findall(v, text)) / n
    sents = [x for x in re.split(r"[.。!！?？]\s", text) if x.strip()]
    out["mean_sent_chars"] = sum(len(x) for x in sents) / max(len(sents), 1)
    out["len_chars"] = len(text)
    out["n_lines"] = text.count("\n") + 1
    out["frac_cjk"] = sum(1 for c in text if "\u4e00" <= c <= "\u9fff") / n
    return out


def cmd_check(a):
    out = Path(a.out)
    files = sorted(out.glob(f"samples_*_{a.tag}.jsonl"))
    if not files:
        raise SystemExit(f"no samples_*_{a.tag}.jsonl in {out}; run `regurgitate.py respond --prompts ...` first")
    by_model = {}
    for p in files:
        model = p.name[len("samples_"):-len(f"_{a.tag}.jsonl")]
        recs = {}
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line); recs[r["pid"]] = r
        by_model[model] = recs
    models = sorted(by_model)
    pids = sorted(set.intersection(*[set(v) for v in by_model.values()]))
    lines = [f"# Math probe: {', '.join(models)}  ({len(pids)} paired problems, tag {a.tag})\n"]

    # accuracy by model x family x lang, and by kind
    acc = defaultdict(lambda: defaultdict(list))
    kind_acc = defaultdict(lambda: defaultdict(list))
    correct = {m: {} for m in models}
    for pid in pids:
        for m in models:
            r = by_model[m][pid]
            pred = parse_answer(r["text"])
            ok = pred is not None and abs(pred - float(r["gold"])) < 1e-6
            correct[m][pid] = ok
            acc[m][(r["family"], r["lang"])].append(ok)
            kind_acc[m][r["kind"]].append(ok)
    lines.append("\n## Accuracy (final answer exactly right)\n\n| model | " + " | ".join(f"{fam}/{lang}" for fam in ["sequence", "control"] for lang in ["zh", "en"]) + " | all |\n|---|" + "---|" * 5 + "\n")
    for m in models:
        cells = [f"{100 * sum(acc[m][(fam, lang)]) / max(len(acc[m][(fam, lang)]), 1):.0f}% (n={len(acc[m][(fam, lang)])})" for fam in ["sequence", "control"] for lang in ["zh", "en"]]
        allv = [v for k in acc[m] for v in acc[m][k]]
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {100 * sum(allv) / max(len(allv), 1):.0f}% |\n")
    lines.append("\nby problem type:\n\n| kind | " + " | ".join(models) + " |\n|---|" + "---|" * len(models) + "\n")
    for kind in sorted({by_model[models[0]][p]["kind"] for p in pids}):
        lines.append(f"| {kind} | " + " | ".join(f"{100 * sum(kind_acc[m][kind]) / max(len(kind_acc[m][kind]), 1):.0f}%" for m in models) + " |\n")

    # paired disagreement
    if len(models) >= 2:
        t, r_ = (a.target if a.target in models else models[0]), (a.ref if a.ref in models else models[-1])
        tw_rr = [p for p in pids if not correct[t][p] and correct[r_][p]]
        tr_rw = [p for p in pids if correct[t][p] and not correct[r_][p]]
        lines.append(f"\n## Paired disagreement ({t} vs {r_})\n- {t} wrong / {r_} right: {len(tw_rr)}  (by family: "
                     + ", ".join(f"{fam}={sum(1 for p in tw_rr if by_model[t][p]['family'] == fam)}" for fam in ["sequence", "control"]) + ")\n"
                     f"- {t} right / {r_} wrong: {len(tr_rw)}  (by family: "
                     + ", ".join(f"{fam}={sum(1 for p in tr_rw if by_model[t][p]['family'] == fam)}" for fam in ["sequence", "control"]) + ")\n")
        # are the wrong answers systematic? (off by d, off by one index, sign, etc.)
        offs = Counter()
        for p in tw_rr[:400]:
            r = by_model[t][p]; pred = parse_answer(r["text"])
            if pred is not None:
                offs[round(pred - float(r["gold"]), 3)] += 1
        lines.append(f"- {t}'s error offsets (pred - gold), most common: {offs.most_common(8)}\n")
        lines.append(f"\n### Examples: {t} wrong, {r_} right\n")
        for p in tw_rr[: a.show]:
            rt, rr = by_model[t][p], by_model[r_][p]
            lines.append(f"\n**{p}** gold={rt['gold']}  {t}→{parse_answer(rt['text'])}  {r_}→{parse_answer(rr['text'])}\n"
                         f"> {t}: {rt['text'][: a.rep_chars].replace(chr(10), ' ⏎ ')}\n>\n> {r_}: {rr['text'][: a.rep_chars].replace(chr(10), ' ⏎ ')}\n")

    # formatting fingerprint
    lines.append("\n## Formatting fingerprint (fraction of responses containing the feature; sequence family vs control)\n\n| feature | "
                 + " | ".join(f"{m} seq | {m} ctrl" for m in models) + " |\n|---|" + "---|" * (2 * len(models)) + "\n")
    F = {m: {fam: [feats(by_model[m][p]["text"]) for p in pids if by_model[m][p]["family"] == fam] for fam in ["sequence", "control"]} for m in models}
    for k in list(FEATURES) + list(STYLE) + ["mean_sent_chars", "len_chars", "n_lines", "frac_cjk"]:
        cells = []
        for m in models:
            for fam in ["sequence", "control"]:
                vals = [f[k] for f in F[m][fam]]
                mean = sum(vals) / max(len(vals), 1)
                cells.append(f"{mean:.2f}" if (k in FEATURES or k in STYLE or k == "frac_cjk") else f"{mean:.0f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |\n")
    lines.append("\n(FEATURES rows = fraction of responses containing it; *_per100 rows = occurrences per 100 characters.)\n")

    # language switching: did the model answer a zh prompt in en or vice versa?
    lines.append("\n## Language of the answer vs language of the prompt (mean CJK fraction of the response)\n")
    for m in models:
        for lang in ["zh", "en"]:
            vals = [feats(by_model[m][p]["text"])["frac_cjk"] for p in pids if by_model[m][p]["lang"] == lang]
            lines.append(f"- {m}, {lang} prompts: {sum(vals) / max(len(vals), 1):.2f}\n")

    txt = "".join(lines)
    print(txt)
    (out / f"mathprobe_{a.tag}.md").write_text(txt)
    print(f"wrote {out / f'mathprobe_{a.tag}.md'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk = sub.add_parser("make"); mk.add_argument("--out", default="math_prompts.jsonl"); mk.add_argument("--n", type=int, default=60); mk.add_argument("--seed", type=int, default=0)
    mk.set_defaults(fn=cmd_make)
    ck = sub.add_parser("check"); ck.add_argument("--out", required=True); ck.add_argument("--tag", default="resp-math_prompts")
    ck.add_argument("--target", default="dolos"); ck.add_argument("--ref", default="qwen"); ck.add_argument("--show", type=int, default=8); ck.add_argument("--rep-chars", type=int, default=500)
    ck.set_defaults(fn=cmd_check)
    a = ap.parse_args(); a.fn(a)


if __name__ == "__main__":
    main()
