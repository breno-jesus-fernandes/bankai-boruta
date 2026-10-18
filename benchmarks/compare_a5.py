"""Local a4/a5/ExtraTrees comparison, also usable for the later Linux campaign.

Run with --a4-python /absolute/a4/env/bin/python --candidate-python /absolute/a5/env/bin/python.
Defaults: industrial 100000 x 500, 20 trees, classic max_iter=20, 2 discarded
warmups and 5 seeds per dtype/thread setting. Every fit gets a fresh process.
Use --correctness for the independent cross-version numerical campaign.
"""
import argparse
from importlib.metadata import version
import itertools
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time


def metadata():
    import psutil
    return dict(platform=platform.platform(), machine=platform.machine(),
        processor=platform.processor(), python=sys.version, cpu_count=os.cpu_count(),
        physical_memory_bytes=psutil.virtual_memory().total,
        versions={p: version(p) for p in ('bankai-boruta', 'bankai-random-forest',
            'numpy', 'scipy', 'scikit-learn', 'Boruta', 'psutil')})


def dataset(n, p, seed, dtype):
    import numpy as np
    rng = np.random.RandomState(seed)
    # Generate once in float32, then promote for exactly equivalent float64 inputs.
    X = np.empty((n, p), dtype=dtype)
    for start in range(0, n, 1024):
        end = min(start + 1024, n)
        X[start:end] = rng.normal(size=(end-start, p)).astype(np.float32)
    signal = X[:, :3].astype(np.float64)
    y = (signal[:, 0] + signal[:, 1] + .5 * signal[:, 2] + rng.normal(scale=.5, size=n) > 0).astype(int)
    return X, y


def selector(config):
    from bankai_boruta import BankaiBoruta
    from bankai_random_forest import BankaiRandomForestClassifier
    common = dict(n_estimators=config['trees'], random_state=config['seed'],
                  max_iter=config['rounds'])
    if config['engine'] == 'extratrees':
        from boruta import BorutaPy
        from sklearn.ensemble import ExtraTreesClassifier
        return BorutaPy(ExtraTreesClassifier(n_estimators=config['trees'], max_depth=6,
            max_features='sqrt', n_jobs=config['threads']), **common)
    return BankaiBoruta(BankaiRandomForestClassifier(n_estimators=config['trees'],
        extra_trees=True, extra_trees_backend='binned', max_bins=63, max_depth=6,
        n_jobs=config['threads']), **common)


def worker(config, output):
    import numpy as np
    import psutil
    X, y = dataset(config['rows'], config['features'], config['seed'], config['dtype'])
    model = selector(config)
    info = metadata()
    baseline = psutil.Process().memory_info().rss
    print(json.dumps(dict(initial_rss_bytes=baseline)), flush=True)
    sys.stdin.readline()  # parent starts its 5ms sampler before releasing the fit
    started = time.perf_counter()
    model.fit(X, y)
    elapsed = time.perf_counter() - started
    diagnostics = getattr(model, 'diagnostics_', {})
    n, p = X.shape
    # a4 has no buffer diagnostics; reconstruct its explicit allocations.
    if config['engine'] == 'a4':
        diagnostics.update(input_dtype=str(X.dtype), native_input_dtype='float64',
            input_conversion_bytes=X.size * 8 if X.dtype == np.float32 else 0,
            permutation_dtype='int64', permutation_bytes=(p + 1) * n * 8)
    Path(output).write_text(json.dumps(dict(config=config, seconds=elapsed,
        initial_rss_bytes=baseline, final_rss_bytes=psutil.Process().memory_info().rss,
        diagnostics=diagnostics, metadata=info,
        n_iter=int(model.n_iter_) if hasattr(model, 'n_iter_') else int(model.importance_history_.shape[0] - 1),
        stop_reason=getattr(model, 'stop_reason_', None),
        support_weak=model.support_weak_.tolist(), ranking=model.ranking_.tolist(),
        selected=np.flatnonzero(model.support_).tolist())))


def run_fit(python, config):
    import psutil
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / 'result.json'
        child = subprocess.Popen([python, str(Path(__file__).resolve()), '--worker',
            json.dumps(config), '--output', str(output)], cwd=tmp,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            env={**os.environ, 'OMP_NUM_THREADS': str(config['threads']),
                 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
        ready = child.stdout.readline()
        if not ready:
            raise RuntimeError(child.stderr.read())
        baseline = json.loads(ready)['initial_rss_bytes']
        peak = baseline
        process = psutil.Process(child.pid)
        child.stdin.write('fit\n')
        child.stdin.flush()
        while child.poll() is None:
            try:
                peak = max(peak, process.memory_info().rss)
            except psutil.NoSuchProcess:
                pass
            time.sleep(.005)
        stdout, stderr = child.communicate()
        if child.returncode:
            raise RuntimeError(stderr)
        result = json.loads(output.read_text())
        peak = max(peak, result['final_rss_bytes'])
        result.update(peak_rss_bytes=peak, incremental_peak_rss_bytes=peak-baseline)
        return result


def correctness_worker(output):
    import numpy as np
    from bankai_boruta import BankaiBoruta
    from bankai_random_forest import BankaiRandomForestClassifier
    results = {}
    for case, (dtype, layout, greedy, classes, features) in enumerate(itertools.product(
            ['float32', 'float64'], ['C', 'F', 'view'], [False, True], [2, 3], [3, 12])):
        X = np.random.RandomState(8).randint(-2, 3, size=(96, features)).astype(dtype)
        X[:, -1] = 0
        X[::2, -1] = -0.
        if layout == 'F': X = np.asfortranarray(X)
        if layout == 'view': X = np.repeat(X, 2, axis=0)[::2, ::-1]
        y = np.arange(len(X)) % classes
        base = BankaiRandomForestClassifier(n_estimators=4, max_depth=3, max_bins=63, n_jobs=1,
            extra_trees=True, extra_trees_backend='binned')
        for fallback in [False, True]:
            model = BankaiBoruta(base, n_estimators=4, max_iter=24, greedy=greedy,
                random_state=7, store_history=True,
                importance_getter='feature_importances_' if fallback else 'auto')
            for refit in [False, True]:
                model.fit(X, y)
                prefix = f'{case}_{fallback}_{refit}_'
                for attr in ['support_', 'support_weak_', 'ranking_', 'importance_history_',
                             'shadow_threshold_history_', 'n_iter_', 'stop_reason_']:
                    results[prefix+attr] = np.asarray(getattr(model, attr))
    # continuous, predictive data: classic must actually shrink its active set.
    X, y = dataset(160, 20, 8, 'float32')
    for fallback in [False, True]:
        model = BankaiBoruta(base, n_estimators=8, max_iter=24, random_state=42,
            store_history=True, importance_getter='feature_importances_' if fallback else 'auto').fit(X, y)
        assert np.isnan(model.importance_history_[-1]).any()
        for attr in ['support_', 'support_weak_', 'ranking_', 'importance_history_',
                     'shadow_threshold_history_', 'n_iter_', 'stop_reason_']:
            results[f'shrink_{fallback}_{attr}'] = np.asarray(getattr(model, attr))
    from sklearn.ensemble import ExtraTreesClassifier
    for scenario in ['weights32', 'weights_int', 'constraints', 'getter', 'external', 'warm_start']:
        X, y = dataset(64, 3, 5, 'float32')
        if scenario == 'weights_int': X = (X * 3).astype(np.int32)
        params = dict(n_estimators=3, max_depth=3, max_bins=63, n_jobs=1,
            extra_trees=True, extra_trees_backend='binned')
        if scenario == 'constraints': params['monotonic_cst'] = [1, 0, 0]
        if scenario == 'warm_start': params['warm_start'] = True
        estimator = BankaiRandomForestClassifier(**params)
        if scenario == 'external':
            estimator = ExtraTreesClassifier(n_estimators=3, max_depth=3, n_jobs=1)
        model = BankaiBoruta(estimator, n_estimators=3, max_iter=4, random_state=7,
            store_history=True, importance_getter=(lambda e: e.feature_importances_) if scenario == 'getter' else 'auto')
        kwargs = {'sample_weight': np.linspace(.5, 1.5, len(X)).astype(np.float32)} if scenario.startswith('weights') else {}
        model.fit(X, y, **kwargs)
        for attr in ['support_', 'support_weak_', 'ranking_', 'importance_history_',
                     'shadow_threshold_history_', 'n_iter_', 'stop_reason_']:
            results[f'{scenario}_{attr}'] = np.asarray(getattr(model, attr))
    np.savez_compressed(output, **results)


def summarize(results):
    groups = {}
    for row in results:
        c = row['config']
        if c['warmup']: continue
        key = (c['dtype'], c['threads'], c['engine'])
        groups.setdefault(key, []).append(row)
    medians = {}
    for key, rows in groups.items():
        medians[key] = dict(dtype=key[0], threads=key[1], engine=key[2], count=len(rows),
            seconds=statistics.median(r['seconds'] for r in rows),
            incremental_peak_rss_bytes=statistics.median(r['incremental_peak_rss_bytes'] for r in rows),
            peak_rss_bytes=statistics.median(r['peak_rss_bytes'] for r in rows))
    # Matching measured Bankai fits must retain selection and round semantics.
    paired = {}
    for row in results:
        c = row['config']
        if c['engine'] in ('a4', 'candidate'):
            paired.setdefault((c['dtype'], c['threads'], c['seed'], c['warmup']), {})[c['engine']] = row
    for pair in paired.values():
        if len(pair) == 2:
            for name in ('selected', 'support_weak', 'ranking', 'n_iter', 'stop_reason'):
                assert pair['a4'][name] == pair['candidate'][name], name
    checks = []
    for dtype, threads in sorted({k[:2] for k in medians}):
        a, b = medians[(dtype, threads, 'a4')], medians[(dtype, threads, 'candidate')]
        reduction = 1-b['incremental_peak_rss_bytes']/a['incremental_peak_rss_bytes']
        ratio = b['seconds']/a['seconds']
        checks.append(dict(dtype=dtype, threads=threads, time_ratio=ratio,
            incremental_rss_reduction=reduction, time_pass=ratio <= 1.05,
            memory_pass=reduction >= .25 if dtype == 'float32' else None))
    return dict(medians=list(medians.values()), checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a4-python')
    parser.add_argument('--candidate-python', default=sys.executable)
    parser.add_argument('--output', default='benchmarks/results/a5-local.json')
    parser.add_argument('--rows', type=int, default=100000)
    parser.add_argument('--features', type=int, default=500)
    parser.add_argument('--trees', type=int, default=20)
    parser.add_argument('--rounds', type=int, default=20)
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 7, 42, 100, 123])
    parser.add_argument('--threads', type=int, nargs='+', default=sorted({1, min(8, os.cpu_count() or 1)}))
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--correctness', action='store_true')
    parser.add_argument('--correctness-worker')
    parser.add_argument('--worker')
    parser.add_argument('--metadata', action='store_true')
    args = parser.parse_args()
    if args.metadata:
        print(json.dumps(metadata()))
        return
    if args.worker:
        worker(json.loads(args.worker), args.output)
        return
    if args.correctness_worker:
        correctness_worker(args.correctness_worker)
        return
    if not args.a4_python: parser.error('--a4-python is required')
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.correctness:
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f'{i}.npz' for i in range(2)]
            for python, path in zip([args.a4_python, args.candidate_python], paths):
                subprocess.run([python, str(Path(__file__).resolve()), '--correctness-worker', str(path)], cwd=tmp, check=True)
            environments = [json.loads(subprocess.check_output([python, str(Path(__file__).resolve()),
                '--metadata'], cwd=tmp, text=True)) for python in [args.a4_python, args.candidate_python]]
            assert environments[0]['python'] == environments[1]['python']
            for package, value in environments[0]['versions'].items():
                if package != 'bankai-boruta':
                    assert value == environments[1]['versions'][package], package
            count = 0
            with np.load(paths[0]) as a, np.load(paths[1]) as b:
                assert a.files == b.files
                for key in a.files:
                    if key.endswith(('importance_history_', 'shadow_threshold_history_')):
                        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-14, equal_nan=True)
                    else: np.testing.assert_array_equal(a[key], b[key])
                    count += 1
            output.write_text(json.dumps(dict(status='passed', compared_arrays=count,
                fits_per_version=count // 7, environments=environments, rtol=0, atol=1e-14), indent=2))
        return
    results = []
    for dtype, threads in itertools.product(['float32', 'float64'], args.threads):
        for trial, seed in enumerate([10000+i for i in range(args.warmups)] + args.seeds):
            engines = ['a4', 'candidate', 'extratrees']
            engines = engines[trial % 3:] + engines[:trial % 3]
            for engine in engines:
                config = dict(rows=args.rows, features=args.features, trees=args.trees,
                    rounds=args.rounds, seed=seed, dtype=dtype, threads=threads,
                    engine=engine, warmup=trial < args.warmups)
                python = args.a4_python if engine == 'a4' else args.candidate_python
                result = run_fit(python, config)
                results.append(result)
                output.write_text(json.dumps(dict(protocol=vars(args), results=results,
                    complete=False), indent=2))
                print(config, round(result['seconds'], 3), result['incremental_peak_rss_bytes'], flush=True)
    output.write_text(json.dumps(dict(protocol=vars(args), results=results,
        complete=True, summary=summarize(results)), indent=2))


if __name__ == '__main__': main()
