"""Reproducible a3 campaign: python benchmarks/validate_a3.py --output ...

Each measurement runs in a fresh subprocess. RSS includes Python, imports,
training storage and native allocations. Sampling can miss short memory peaks.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time


def worker(config):
    import numpy as np
    import psutil
    from bankai_boruta import BankaiBoruta
    from bankai_random_forest import BankaiRandomForestClassifier
    from boruta import BorutaPy
    from greedyboruta import GreedyBorutaPy
    n, p, seed, engine, greedy = config
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n, p))
    X[:, 3] = X[:, 0] + rng.normal(scale=.1, size=n)
    y = (X[:, 0] + X[:, 1] + .5 * X[:, 2] + rng.normal(scale=.5, size=n) > 0).astype(int)
    forest = BankaiRandomForestClassifier(n_estimators=20, extra_trees=True,
        extra_trees_backend='binned', max_bins=63, max_depth=6, n_jobs=-1)
    common = dict(n_estimators=20, random_state=seed)
    if engine in ('rust', 'adapter'):
        selector = BankaiBoruta(forest, greedy=greedy, max_iter=20, store_history=True,
            importance_getter='auto' if engine == 'rust' else lambda e: e.feature_importances_, **common)
    elif engine == 'borutapy':
        selector = BorutaPy(forest, max_iter=20, **common)
    else:
        selector = GreedyBorutaPy(forest, n_jobs=1, **common)
    process = psutil.Process()
    peak = [process.memory_info().rss]
    done = threading.Event()
    def sample():
        while not done.wait(.002):
            peak[0] = max(peak[0], process.memory_info().rss)
    sampler = threading.Thread(target=sample)
    sampler.start()
    started = time.perf_counter()
    try:
        selector.fit(X, y)
    finally:
        elapsed = time.perf_counter() - started
        peak[0] = max(peak[0], process.memory_info().rss)
        done.set()
        sampler.join()
    result = dict(n=n, p=p, seed=seed, engine=engine, greedy=greedy,
        seconds=elapsed, peak_rss_bytes=peak[0],
        recall=float(selector.support_[:4].mean()), false_positive_rate=float(selector.support_[4:].mean()),
        selected=np.flatnonzero(selector.support_).tolist(), diagnostics=getattr(selector, 'diagnostics_', {}))
    if engine in ('rust', 'adapter'):
        result['history'] = [[float(v) if np.isfinite(v) else None for v in row] for row in selector.importance_history_]
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--output', default='benchmarks/results/a3.json')
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(json.loads(args.worker))))
        return
    import psutil
    results = []
    for n, p in [(200, 10), (1000, 30)]:
        for seed in [1, 7, 42]:
            for engine, greedy in [('rust', False), ('adapter', False), ('borutapy', False), ('rust', True), ('adapter', True), ('greedyboruta', True)]:
                config = [n, p, seed, engine, greedy]
                child = subprocess.Popen([sys.executable, __file__, '--worker', json.dumps(config)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                monitored = psutil.Process(child.pid)
                peak = 0
                while child.poll() is None:
                    try:
                        peak = max(peak, monitored.memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                    time.sleep(.002)
                stdout, stderr = child.communicate()
                if child.returncode:
                    raise RuntimeError(stderr)
                results.append(json.loads(stdout))
                # Parent sampling continues while native code holds the child GIL.
                results[-1]['peak_rss_bytes'] = max(peak, results[-1]['peak_rss_bytes'])
                results[-1].pop('incremental_peak_rss_bytes', None)
                print(config, round(results[-1]['seconds'], 3), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    import platform
    from importlib.metadata import version
    output.write_text(json.dumps(dict(platform=platform.platform(), python=sys.version, cpu_count=os.cpu_count(), versions={name: version(name) for name in ('bankai-boruta', 'bankai-random-forest', 'Boruta', 'greedyboruta', 'numpy', 'scikit-learn')}, results=results), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
