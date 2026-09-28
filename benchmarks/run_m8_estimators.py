"""Paired 10k-row benchmarks for native and external-estimator M8 paths.

Every measured method runs in a fresh subprocess. This report is a bounded
performance comparison, not evidence of selection quality or convergence.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BENCHMARK_ROWS = 10_000
BENCHMARK_FEATURES = 100
METHODS = (
    "bankai_native",
    "bankai_external_xrf",
    "bankai_external_sklearn",
    "borutapy_sklearn",
)


def validate_dimensions(rows: int, features: int) -> None:
    if rows > BENCHMARK_ROWS:
        raise ValueError("M8 benchmark protocol is capped at 10.000 rows")
    if rows < 1 or features < 1:
        raise ValueError("rows and features must be positive")


def sklearn_forest_parameters(trees: int, seed: int, features: int) -> dict:
    del features  # sqrt selection is expressed identically by sklearn and BorutaPy.
    return {
        "n_estimators": trees,
        "max_depth": 16,
        "max_features": "sqrt",
        "n_jobs": 1,
        "class_weight": "balanced",
        "random_state": seed,
    }


def _worker(args) -> dict:
    import numpy as np
    import psutil
    import sklearn
    from benchmarks.datasets import make_boruta_case

    case = make_boruta_case(
        "mixed", seed=args.seed, n_samples=args.rows, n_features=args.features
    )
    X, y = case.X, case.y
    process = psutil.Process()
    rss_before = 0
    selected = None
    n_iter = None
    copied_bytes = 0
    callback_count = 0
    callback_seconds = 0.0
    estimator_config = None
    started = 0.0

    if args.method == "bankai_native":
        from bankai_boruta import BankaiBoruta

        model = BankaiBoruta(
            n_estimators=args.trees,
            max_iter=args.iterations,
            alpha=0.01,
            max_depth=16,
            max_features="sqrt",
            n_jobs=1,
            random_state=args.seed,
        )
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        model.fit(X, y)
        selected = np.flatnonzero(model.support_)
        n_iter = model.n_iter_
        copied_bytes = model.diagnostics_["copied_bytes"]
    elif args.method in {"bankai_external_sklearn", "borutapy_sklearn"}:
        from sklearn.ensemble import RandomForestClassifier

        estimator_config = sklearn_forest_parameters(args.trees, args.seed, args.features)
        estimator = RandomForestClassifier(**estimator_config)
        if args.method == "bankai_external_sklearn":
            from bankai_boruta import BankaiBoruta

            model = BankaiBoruta(
                estimator=estimator,
                importance_getter="auto",
                n_estimators=args.trees,
                max_iter=args.iterations,
                alpha=0.01,
                max_depth=16,
                max_features="sqrt",
                n_jobs=1,
                random_state=args.seed,
            )
            rss_before = process.memory_info().rss
            started = time.perf_counter()
            model.fit(X, y)
            selected = np.flatnonzero(model.support_)
            n_iter = model.n_iter_
            copied_bytes = model.diagnostics_["copied_bytes"]
            callback_count = model.diagnostics_["external_estimator_calls"]
            callback_seconds = model.diagnostics_["external_estimator_seconds"]
        else:
            from boruta import BorutaPy

            model = BorutaPy(
                estimator,
                n_estimators=args.trees,
                max_iter=args.iterations + 1,
                alpha=0.01,
                perc=100,
                two_step=False,
                random_state=args.seed,
                verbose=0,
            )
            rss_before = process.memory_info().rss
            started = time.perf_counter()
            model.fit(X, y)
            selected = np.flatnonzero(model.support_)
            n_iter = int(model.importance_history_.shape[0] - 1)
    elif args.method == "bankai_external_xrf":
        if args.xrf_path:
            sys.path.insert(0, str(Path(args.xrf_path).expanduser().resolve()))
        from bankai_random_forest import BankaiRandomForestClassifier
        from bankai_boruta import BankaiBoruta

        estimator_config = {
            "n_estimators": args.trees,
            "max_depth": 16,
            "max_features": "sqrt",
            "n_jobs": 1,
            "random_state": args.seed,
            "class_weight": "balanced",
            "importance_type": "permutation",
        }
        def nonnegative_rank_preserving_importance(fitted):
            scores = np.asarray(fitted.feature_importances_, dtype=float)
            return scores - scores.min()

        model = BankaiBoruta(
            estimator=BankaiRandomForestClassifier(**estimator_config),
            importance_getter=nonnegative_rank_preserving_importance,
            n_estimators=args.trees,
            max_iter=args.iterations,
            alpha=0.01,
            max_depth=16,
            max_features="sqrt",
            n_jobs=1,
            random_state=args.seed,
        )
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        model.fit(X, y)
        selected = np.flatnonzero(model.support_)
        n_iter = model.n_iter_
        copied_bytes = model.diagnostics_["copied_bytes"]
        callback_count = model.diagnostics_["external_estimator_calls"]
        callback_seconds = model.diagnostics_["external_estimator_seconds"]
    else:
        raise ValueError(f"unknown method {args.method}")

    fit_seconds = time.perf_counter() - started
    rss_after = process.memory_info().rss
    result = {
        "method": args.method,
        "seed": args.seed,
        "rows": args.rows,
        "features": args.features,
        "trees": args.trees,
        "iteration_limit": args.iterations,
        "n_iter": n_iter,
        "fit_seconds": fit_seconds,
        "rss_delta_bytes": rss_after - rss_before,
        "copied_bytes": copied_bytes,
        "external_estimator_calls": callback_count,
        "external_estimator_seconds": callback_seconds,
        "selected_count": int(len(selected)),
        "estimator_config": estimator_config,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "python": platform.python_version(),
    }
    print(json.dumps(result, sort_keys=True))
    return result


def _worker_command(args, method: str, seed: int) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--method",
        method,
        "--seed",
        str(seed),
        "--rows",
        str(args.rows),
        "--features",
        str(args.features),
        "--trees",
        str(args.trees),
        "--iterations",
        str(args.iterations),
        "--xrf-path",
        str(args.xrf_path or ""),
    ]


def _summarize(records: list[dict]) -> list[dict]:
    groups = {}
    for record in records:
        if record["phase"] == "warmup":
            continue
        groups.setdefault(record["method"], []).append(record)
    summary = []
    for method, values in sorted(groups.items()):
        summary.append({
            "method": method,
            "runs": len(values),
            "median_fit_seconds": statistics.median(item["fit_seconds"] for item in values),
            "median_rss_delta_mib": statistics.median(item["rss_delta_bytes"] for item in values) / (1024 * 1024),
            "median_copied_bytes": statistics.median(item["copied_bytes"] for item in values),
            "median_external_estimator_calls": statistics.median(item["external_estimator_calls"] for item in values),
            "median_external_estimator_seconds": statistics.median(item["external_estimator_seconds"] for item in values),
            "median_selected_count": statistics.median(item["selected_count"] for item in values),
            "median_n_iter": statistics.median(item["n_iter"] for item in values if item["n_iter"] is not None),
        })
    return summary


def _markdown(result: dict) -> str:
    metadata = result["metadata"]
    lines = [
        "# M8 — Benchmarks de estimadores externos (10k×100)",
        "",
        f"Execução: {metadata['date']} · {metadata['platform']} · Python {metadata['python']}",
        f"scikit-learn {metadata['scikit_learn']} · NumPy {metadata['numpy']}",
        "",
        "## Protocolo",
        "",
        "- Todas as execuções usam somente 10.000×100, dados sintéticos mistos e processos isolados.",
        f"- {metadata['repeats']} repetições medidas após um aquecimento por método; {metadata['trees']} árvores, até {metadata['iterations']} iterações, profundidade 16 e uma thread.",
        "- `bankai_external_sklearn` versus `borutapy_sklearn` é o par justo: ambos usam o mesmo RandomForest sklearn, importância impurity, seed e configuração. Muda o controller e a materialização do adapter.",
        "- `bankai_native` usa o provider Rust/XRF e importância permutation; compara o caminho default completo, não isola somente o overhead do adapter.",
        "- `bankai_external_xrf` usa a importância permutation do BankaiRandomForestClassifier. O getter subtrai o mínimo global do vetor por iteração, preservando a ordem real-vs-shadow e tornando os scores não negativos para o contrato do selector.",
        "- Com duas iterações, os resultados não permitem concluir qualidade, convergência ou liderança universal.",
        "- RSS é delta do processo antes/depois do fit, não pico de memória. Bytes copiados contam a matriz aumentada conhecida; não incluem alocações internas do estimator.",
        "",
        "## Resultados",
        "",
        "| Método | Runs | Mediana fit (s) | RSS delta mediano (MiB) | Bytes X copiados | Callbacks medianos | Tempo callback (s) | Selecionadas | Iterações |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        lines.append(
            f"| {row['method']} | {row['runs']} | {row['median_fit_seconds']:.4f} | "
            f"{row['median_rss_delta_mib']:.2f} | {row['median_copied_bytes']:.0f} | "
            f"{row['median_external_estimator_calls']:.1f} | {row['median_external_estimator_seconds']:.4f} | "
            f"{row['median_selected_count']:.1f} | {row['median_n_iter']:.1f} |"
        )
    lines += [
        "",
        "Os resultados por repetição, configurações e revisões estão em [`m8_estimators.json`](m8_estimators.json).",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--rows", type=int, default=BENCHMARK_ROWS)
    parser.add_argument("--features", type=int, default=BENCHMARK_FEATURES)
    parser.add_argument("--trees", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--xrf-path", default="")
    parser.add_argument("--bankai-revision", default="unknown")
    parser.add_argument("--xrf-revision", default="unknown")
    parser.add_argument("--output-json", type=Path, default=ROOT / "benchmarks/results/m8_estimators.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "benchmarks/results/m8_estimators.md")
    args = parser.parse_args()
    try:
        validate_dimensions(args.rows, args.features)
    except ValueError as error:
        parser.error(str(error))
    if min(args.trees, args.iterations, args.repeats) < 1:
        parser.error("trees, iterations, and repeats must be positive")
    if args.worker:
        if not args.method:
            parser.error("--method is required in worker mode")
        _worker(args)
        return

    methods = list(args.methods or METHODS)
    records = []
    for phase, repeat in [("warmup", 0), *(('performance', i) for i in range(args.repeats))]:
        order = list(methods)
        random.Random(1_000 + repeat).shuffle(order)
        for method in order:
            seed = args.seed + repeat
            raw = subprocess.check_output(
                _worker_command(args, method, seed), cwd="/tmp", text=True
            )
            record = json.loads(raw.strip().splitlines()[-1])
            record["phase"] = phase
            record["repeat"] = repeat
            records.append(record)
            print(f"{phase}: {method} {record['fit_seconds']:.3f}s", flush=True)

    import importlib.metadata as md
    metadata = {
        "date": time.strftime("%Y-%m-%d"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "rows": args.rows,
        "features": args.features,
        "trees": args.trees,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "methods": methods,
        "bankai_boruta_revision": args.bankai_revision,
        "xrf_revision": args.xrf_revision,
        "borutapy_version": md.version("Boruta") if "borutapy_sklearn" in methods else None,
    }
    result = {"metadata": metadata, "summary": _summarize(records), "runs": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(f"wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
