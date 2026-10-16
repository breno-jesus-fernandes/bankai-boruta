"""Validate recorded parity and print aggregate campaign tables."""
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else 'benchmarks/results/a3-final.json')
campaign = json.loads(path.read_text())
rows = campaign['results']
maximum = 0.
for row in rows:
    if row['engine'] != 'rust':
        continue
    other = next(r for r in rows if r['engine'] == 'adapter' and all(r[k] == row[k] for k in ('n', 'p', 'seed', 'greedy')))
    a, b = np.asarray(row['history'], dtype=float), np.asarray(other['history'], dtype=float)
    np.testing.assert_allclose(a, b, atol=1e-12, rtol=0, equal_nan=True)
    assert row['selected'] == other['selected']
    maximum = max(maximum, float(np.nanmax(np.abs(a-b))))
print(f'Maximum absolute Rust/adapter importance difference: {maximum:.3g}; every paired decision agrees.\n')
groups = defaultdict(list)
for row in rows:
    groups[(row['n'], row['p'], row['greedy'], row['engine'])].append(row)
print('| Rows × columns | Mode | Engine | Median ms | Median peak RSS MiB | Mean recall | Mean noise FPR |')
print('|---|---|---|---:|---:|---:|---:|')
for (n, p, greedy, engine), group in groups.items():
    print(f'| {n} × {p} | {"greedy" if greedy else "classic"} | {engine} | {np.median([r["seconds"] for r in group])*1000:.2f} | {np.median([r["peak_rss_bytes"] for r in group])/2**20:.2f} | {np.mean([r["recall"] for r in group]):.3f} | {np.mean([r["false_positive_rate"] for r in group]):.3f} |')
print('\n| Rows × columns | Mode | Engine | Preparation ms | Training ms | Augmentation copy ms | Augmented float bytes |')
print('|---|---|---|---:|---:|---:|---:|')
for (n, p, greedy, engine), group in groups.items():
    if engine not in ('rust', 'adapter'):
        continue
    metrics = [np.median([r['diagnostics'][key] for r in group]) for key in ('preparation_seconds', 'training_seconds', 'copy_seconds', 'copied_bytes')]
    print(f'| {n} × {p} | {"greedy" if greedy else "classic"} | {engine} | {metrics[0]*1000:.2f} | {metrics[1]*1000:.2f} | {metrics[2]*1000:.2f} | {metrics[3]:.0f} |')
