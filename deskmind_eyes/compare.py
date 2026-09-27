"""Paired comparison of two ScreenSpot eval runs (same samples, per-sample hit/miss).

python -m deskmind_eyes.compare runs/eval/base.jsonl \
    runs/eval/sft_step50.jsonl

Averages alone mislead: the same checkpoint re-evaluated at temperature 0 moves by
~0.7 points. McNemar's test only counts samples that flipped between the two runs,
so it separates a real change from sampling noise. Rule of thumb used here:
|z| >= 2 -> "significant", otherwise report the numbers without a conclusion.
"""

import argparse
import json
import math
from collections import defaultdict

SIGNIFICANT_Z = 2.0


def load(path: str) -> dict[tuple[str, str], dict]:
    return {(r["img_filename"], r["instruction"]): r for r in map(json.loads, open(path))}


def mcnemar_z(gained: int, lost: int) -> float:
    return (gained - lost) / math.sqrt(gained + lost) if gained + lost else 0.0


def compare(a: dict, b: dict) -> dict:
    keys = sorted(a.keys() & b.keys())
    if len(keys) != len(a) or len(keys) != len(b):
        raise SystemExit(f"sample sets differ: {len(a)} vs {len(b)}, {len(keys)} shared")

    groups = defaultdict(list)
    for k in keys:
        groups[f"{a[k].get('group', a[k].get('device'))}-{a[k]['data_type']}"].append(k)
    groups["avg"] = keys

    rows = {}
    for name, ks in sorted(groups.items(), key=lambda kv: (kv[0] == "avg", kv[0])):
        gained = sum(1 for k in ks if b[k]["hit"] and not a[k]["hit"])
        lost = sum(1 for k in ks if a[k]["hit"] and not b[k]["hit"])
        acc_a = 100 * sum(a[k]["hit"] for k in ks) / len(ks)
        acc_b = 100 * sum(b[k]["hit"] for k in ks) / len(ks)
        z = mcnemar_z(gained, lost)
        rows[name] = dict(n=len(ks), a=round(acc_a, 1), b=round(acc_b, 1),
                          delta=round(acc_b - acc_a, 1), gained=gained, lost=lost,
                          z=round(z, 2), significant=abs(z) >= SIGNIFICANT_Z)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="reference run (e.g. base model)")
    ap.add_argument("b", help="candidate run (e.g. checkpoint)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    rows = compare(load(args.a), load(args.b))
    if args.json:
        print(json.dumps(rows, ensure_ascii=False))
        return
    print(f"A = {args.a}\nB = {args.b}\n")
    print(f"{'group':14s} {'n':>5s} {'A':>6s} {'B':>6s} {'Δ':>6s} {'+':>4s} {'-':>4s} {'z':>6s}")
    for name, r in rows.items():
        flag = "  *" if r["significant"] else ""
        print(f"{name:14s} {r['n']:5d} {r['a']:6.1f} {r['b']:6.1f} {r['delta']:+6.1f} "
              f"{r['gained']:4d} {r['lost']:4d} {r['z']:6.2f}{flag}")
    verdict = rows["avg"]
    if verdict["significant"]:
        print(f"\navg: {'better' if verdict['z'] > 0 else 'worse'} (|z| >= {SIGNIFICANT_Z})")
    else:
        print(f"\navg: no significant difference (|z| < {SIGNIFICANT_Z}); treat Δ as noise")


if __name__ == "__main__":
    main()
