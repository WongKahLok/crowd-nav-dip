"""Compare two evaluation arms episode by episode.

Both arms must have been run with the same seeds, maps and densities (run_arm.sh does
this), so every episode in one arm has a twin in the other with the same scenario.
Standard library only.

    python crowd_nav/compare_arms.py \
        --base crowd_nav/results-v2/b4_theta_sarl_*.csv \
        --variant crowd_nav/results-v2/lookahead_fixed1_*.csv

Reports, per map and density: outcome counts for both arms; then, over the paired
episodes, which outcomes changed, an exact McNemar test on success vs failure, and
the paired differences of the motion metrics.
"""
import argparse
import csv
import math
import statistics as st

METRICS = ['steps', 'sim_time', 'detour_ratio', 'min_sep_dynamic', 'min_sep_static',
           'danger_frac', 'emergency_frac']


def load(paths):
    rows = {}
    for path in paths:
        with open(path, newline='') as fh:
            for r in csv.DictReader(fh):
                key = (r['map'], r['requested_n_dynamic'], int(r['seed']))
                rows[key] = r
    return rows


def number(row, field):
    value = row.get(field, '')
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mcnemar(base, variant, keys):
    """Exact two-sided McNemar on success vs not-success over the paired episodes."""
    b = sum(base[k]['outcome'] == 'success' and variant[k]['outcome'] != 'success' for k in keys)
    c = sum(base[k]['outcome'] != 'success' and variant[k]['outcome'] == 'success' for k in keys)
    n = b + c
    if n == 0:
        return b, c, 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return b, c, min(1.0, 2 * tail / 2 ** n)


def main():
    ap = argparse.ArgumentParser(description='Compare two evaluation arms seed by seed')
    ap.add_argument('--base', nargs='+', required=True, help='CSV files of the reference arm')
    ap.add_argument('--variant', nargs='+', required=True, help='CSV files of the arm being tested')
    ap.add_argument('--labels', nargs=2, default=['base', 'variant'])
    ap.add_argument('--show', type=int, default=10, help='how many changed episodes to list')
    args = ap.parse_args()

    A, B = load(args.base), load(args.variant)
    keys = sorted(set(A) & set(B), key=lambda k: (k[0], int(k[1]), k[2]))
    la, lb = args.labels
    print(f'{la}: {len(A)} episodes | {lb}: {len(B)} episodes | paired on the same scenario: {len(keys)}')
    if not keys:
        raise SystemExit('no episodes share a map, density and seed: were both arms run the same way?')
    unpaired = (len(A) - len(keys)) + (len(B) - len(keys))
    if unpaired:
        print(f'  note: {unpaired} episode(s) had no twin and are ignored')

    print(f"\n{'map':5s} {'ships':>5s} {'n':>4s}  {la + ' succ/coll/time':>22s}  {lb + ' succ/coll/time':>22s}")
    cells = {}
    for k in keys:
        cells.setdefault((k[0][12:14], k[1]), []).append(k)
    for (m, d), ks in sorted(cells.items(), key=lambda kv: (kv[0][0], int(kv[0][1]))):
        line = f'{m:5s} {d:>5s} {len(ks):4d}  '
        for R in (A, B):
            s = sum(R[k]['outcome'] == 'success' for k in ks)
            c = sum(R[k]['outcome'].startswith('collision') for k in ks)
            t = sum(R[k]['outcome'] == 'timeout' for k in ks)
            line += f'{s:8d} /{c:4d} /{t:5d}  '
        print(line)

    changed = [k for k in keys if A[k]['outcome'] != B[k]['outcome']]
    print(f'\noutcome changed in {len(changed)} of {len(keys)} paired episodes')
    for k in changed[:args.show]:
        print(f"  {k[0][12:14]} {k[1]:>2s} ships seed {k[2]:3d}: {A[k]['outcome']:18s} -> {B[k]['outcome']}")
    if len(changed) > args.show:
        print(f'  ... and {len(changed) - args.show} more')

    b, c, p = mcnemar(A, B, keys)
    verdict = 'no measurable difference' if p > 0.05 else f'{lb} differs from {la}'
    print(f'success rate: {la}-only successes {b}, {lb}-only successes {c}, '
          f'exact McNemar p = {p:.3f} -> {verdict}')

    both = [k for k in keys if A[k]['outcome'] == 'success' and B[k]['outcome'] == 'success']
    print(f'\npaired differences ({lb} minus {la}) over the {len(both)} episodes both arms completed:')
    for field in METRICS:
        diffs = [number(B[k], field) - number(A[k], field) for k in both
                 if number(A[k], field) is not None and number(B[k], field) is not None]
        if not diffs:
            continue
        same = sum(abs(d) < 1e-9 for d in diffs)
        print(f'  {field:16s} mean {st.mean(diffs):+9.3f}  median {st.median(diffs):+9.3f}  '
              f'unchanged in {same}/{len(diffs)}')


if __name__ == '__main__':
    main()
