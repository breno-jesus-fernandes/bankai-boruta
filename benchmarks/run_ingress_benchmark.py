"""M6 ingress/regression gate; compare installed pre-refactor and current engines.

Run from the repository with --baseline-python pointing to an isolated venv
containing the pre-M6 wheel. Each worker warms up once, shuffles layouts per
repeat, and times only fit. RSS is process high-water including all fixture
layouts, imports and warm-up, NOT an isolated per-layout allocation measure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


def worker(args):
    import numpy as np
    import sklearn
    from bankai_boruta import BankaiBoruta, _core

    n, p = args.shape
    rng = np.random.default_rng(173)
    y = rng.integers(0, 2, n)
    values = rng.normal(size=(n, p))
    values[:, 0] = y + rng.normal(0, 0.1, n)
    if args.dtype == 'f32':
        values = values.astype('float32')
    elif args.dtype == 'mixed':
        values[:, ::2] = values[:, ::2].astype('float32')
    layouts = {'numpy_c': np.ascontiguousarray(values), 'numpy_f': np.asfortranarray(values)}
    setup_seconds = {}
    if args.current:
        import polars as pl
        def make_frame(v):
            return pl.DataFrame({f'x{i}': v[:, i].astype('float32')
                                if args.dtype == 'mixed' and i % 2 == 0 else v[:, i]
                                for i in range(p)})
        for chunks in [1, 8, 64]:
            started = time.perf_counter()
            boundaries = np.linspace(0, n, chunks+1, dtype=int)
            layouts[f'polars_{chunks}'] = pl.concat(
                [make_frame(values[a:b]) for a, b in zip(boundaries[:-1], boundaries[1:])], rechunk=False)
            setup_seconds[f'polars_{chunks}'] = time.perf_counter()-started
        padded = np.concatenate([values[:1], values, values[-1:]])
        layouts['polars_slice'] = make_frame(padded).slice(1, n)
        for name, X in layouts.items():
            assert np.array_equal(np.asarray(X), values), name
    options = dict(n_estimators=16, max_iter=3, max_depth=16, n_jobs=args.jobs,
                   random_state=29, store_history=True)
    BankaiBoruta(**options).fit(layouts['numpy_c'], y)  # warm-up outside timing
    records = []
    for repeat in range(args.repeats):
        names = list(layouts)
        random.Random(41+repeat).shuffle(names)
        for name in names:
            X = layouts[name]
            before = time.perf_counter()
            if name.startswith('polars'):
                _core.inspect_arrow(X)
            else:
                _core.inspect_matrix(X)
            ingress = time.perf_counter()-before
            before = time.perf_counter()
            model = BankaiBoruta(**options).fit(X, y)
            elapsed = time.perf_counter()-before
            signature = hashlib.sha256(json.dumps([model.decision_.tolist(), model.history_]).encode()).hexdigest()
            records.append(dict(layout=name, repeat=repeat, seconds=elapsed,
                                ingress_validation_seconds=ingress, signature=signature,
                                copied_bytes=model.diagnostics_['copied_bytes']))
    assert len({v['signature'] for v in records}) == 1, 'layout changed decisions/hits'
    return dict(shape=args.shape, dtype=args.dtype, jobs=args.jobs, current=args.current,
                setup_seconds=setup_seconds, records=records,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == 'darwin' else 1024),
                numpy=np.__version__, sklearn=sklearn.__version__, python=platform.python_version())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--current', action='store_true')
    parser.add_argument('--shape', nargs=2, type=int, default=[1000, 20])
    parser.add_argument('--dtype', default='f64')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--baseline-python', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args)))
        return
    if not args.baseline_python or not args.output:
        parser.error('--baseline-python and --output are required')
    runs, ratios = [], []
    for shape in [(1000, 20), (10000, 100)]:
        for dtype in ['f32', 'f64', 'mixed']:
            for jobs in [1, 4]:
                pair = {}
                order = [False, True] if jobs == 1 else [True, False]
                for current in order:
                    command = [sys.executable if current else str(args.baseline_python),
                               str(Path(__file__).resolve()), '--worker', '--shape', *map(str, shape),
                               '--dtype', dtype, '--jobs', str(jobs), '--repeats', str(args.repeats)]
                    if current:
                        command.append('--current')
                    pair[current] = json.loads(subprocess.check_output(command, cwd='/tmp', text=True))
                    runs.append(pair[current])
                    print(f'{shape} {dtype} jobs={jobs} current={current} done', flush=True)
                assert pair[False]['records'][0]['signature'] == pair[True]['records'][0]['signature'], 'refactor changed hits/decisions'
                for layout in ['numpy_c', 'numpy_f']:
                    means = {key: statistics.mean(r['seconds'] for r in value['records'] if r['layout'] == layout)
                             for key, value in pair.items()}
                    ratios.append(dict(shape=shape, dtype=dtype, jobs=jobs, layout=layout,
                                       current_over_baseline=means[True]/means[False]))
                # Durable progress if later configurations fail.
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(dict(platform=platform.platform(), repeats=args.repeats,
                    n_estimators=16, max_iter=3, max_depth=16, seed=29,
                    baseline_revision='a5944cd (same runtime as pre-M6 wheel)',
                    runs=runs, ratios=ratios,
                    numpy_geomean_ratio=math.exp(statistics.mean(math.log(r['current_over_baseline']) for r in ratios))), indent=2)+'\n')


if __name__ == '__main__':
    main()
