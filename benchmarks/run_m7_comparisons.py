"""Run bounded, reproducible Boruta implementation comparisons (<=10k rows).

Each measured case runs in a fresh subprocess so max RSS is not contaminated
by earlier implementations. Timings exclude process startup and data creation.
The harness pins estimator settings but does not claim algorithmic equivalence
for SHAP or the Python orchestration diagnostic.
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
METHODS = (
    "sklearn_rf",
    "bankai_xrf_rf",
    "bankai_boruta",
    "borutapy",
    "borutapy_bankai_xrf",
    "boruta_fru",
    "borutashap_gini",
    "borutashap_shap",
    "python_orchestration_xrf",
)
SELECTORS = ("bankai_boruta", "borutapy", "borutapy_bankai_xrf", "boruta_fru", "borutashap_gini", "borutashap_shap")
QUALITY_SEEDS = (11, 29, 47, *random.Random(20_260_927).sample(range(1_000, 1_000_000), 27))
BENCHMARK_ROWS = 10_000
BENCHMARK_FEATURES = 100
PERFORMANCE_DIMENSIONS = ((BENCHMARK_ROWS, BENCHMARK_FEATURES),)


def _case(scenario: str, seed: int, n: int, p: int):
    from benchmarks.datasets import make_boruta_case

    case = make_boruta_case(scenario, seed=seed, n_samples=n, n_features=p)
    return case.X, case.y, case.relevant_features


def _score(selected, relevant, n_features):
    selected = sorted(set(int(i) for i in selected))
    tp = len(set(selected) & set(relevant))
    precision = tp / len(selected) if selected else (1.0 if not relevant else 0.0)
    recall = tp / len(relevant) if relevant else (1.0 if not selected else 0.0)
    return {
        "selected": selected,
        "selected_count": len(selected),
        "precision": precision,
        "recall": recall,
        "false_positives": len(set(selected) - set(relevant)),
    }


def worker(args):
    import numpy as np
    import psutil
    import sklearn
    from sklearn.ensemble import RandomForestClassifier

    X, y, relevant = _case(args.scenario, args.seed, args.rows, args.features)
    names = [f"f{i}" for i in range(args.features)]
    selected = None
    n_iter = None
    extra = {}
    setup_seconds = 0.0
    process = psutil.Process()

    if args.method == "sklearn_rf":
        model = RandomForestClassifier(
            n_estimators=args.trees, max_depth=16, max_features="sqrt",
            n_jobs=1, random_state=args.seed, oob_score=True,
        )
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        model.fit(X, y)
        scores = _score(np.argsort(model.feature_importances_)[-2:], relevant, args.features)
        selected = scores["selected"]
        extra["comparison_role"] = "forest-only reference; not a Boruta selector"
    elif args.method == "bankai_xrf_rf":
        from bankai_random_forest import BankaiRandomForestClassifier

        model = BankaiRandomForestClassifier(
            n_estimators=args.trees, max_depth=16, max_features="sqrt",
            n_jobs=1, random_state=args.seed, oob_score=False,
            importance_type="permutation",
        )
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        model.fit(X, y)
        scores = _score(np.argsort(model.feature_importances_)[-2:], relevant, args.features)
        selected = scores["selected"]
        extra["comparison_role"] = "Bankai XRF-fork forest plus native OOB permutation importance; not Boruta"
        extra["engine_revision"] = args.xrf_revision
    elif args.method == "bankai_boruta":
        from bankai_boruta import BankaiBoruta

        model = BankaiBoruta(
            n_estimators=args.trees, max_iter=args.iterations, alpha=0.01,
            max_depth=16, max_features="sqrt", n_jobs=1,
            random_state=args.seed, store_history=False,
        )
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        model.fit(X, y)
        selected = np.flatnonzero(model.support_)
        n_iter = model.n_iter_
        extra.update({"stop_reason": model.stop_reason_, "copied_bytes": model.diagnostics_["copied_bytes"]})
    elif args.method in {"borutapy", "borutapy_bankai_xrf"}:
        from boruta import BorutaPy

        if args.method == "borutapy_bankai_xrf":
            from bankai_random_forest import BankaiRandomForestClassifier

            estimator = BankaiRandomForestClassifier(
                n_estimators=args.trees, max_depth=16, max_features="sqrt",
                n_jobs=1, random_state=args.seed, class_weight="balanced",
                importance_type="gain",
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=args.trees, max_depth=16, max_features="sqrt",
                n_jobs=1, random_state=args.seed, class_weight="balanced",
            )
        model = BorutaPy(
            # BorutaPy starts its counter at 1 and loops while iter < max_iter.
            # Add one to execute the same requested number of iterations as
            # native engines whose counter starts at zero.
            estimator, n_estimators=args.trees, max_iter=args.iterations + 1,
            alpha=0.01, perc=100, two_step=False, random_state=args.seed, verbose=0,
        )
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        model.fit(X, y)
        selected = np.flatnonzero(model.support_)
        n_iter = int(model.importance_history_.shape[0] - 1)
        extra["tentative_count"] = int(np.sum(model.support_weak_))
        if args.method == "borutapy_bankai_xrf":
            extra.update({
                "comparison_role": "same BorutaPy controller/settings as sklearn baseline, with Bankai XRF forest and Gini gain importance",
                "engine_revision": args.xrf_revision,
            })
    elif args.method == "boruta_fru":
        import polars as pl
        from boruta_fru import Boruta

        setup_started = time.perf_counter()
        frame = pl.DataFrame({name: X[:, i] for i, name in enumerate(names)})
        target = pl.Series("target", np.where(y == 1, "class1", "class0")).cast(pl.Categorical)
        setup_seconds = time.perf_counter() - setup_started
        model = Boruta(
            max_runs=args.iterations, trees=args.trees,
            tries=max(1, int(np.sqrt(args.features))), impute=False,
            seed=args.seed, threads=1,
        )
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        model.fit(frame, target)
        decisions = np.asarray(model.final_decision()).astype(str)
        selected = np.flatnonzero(np.char.lower(decisions) == "confirmed")
        n_iter = None  # boruta-fru's public API does not expose actual runs.
        extra.update({
            "decision_values": sorted(set(decisions.tolist())),
            "actual_iteration_count_exposed": False,
        })
    elif args.method in {"borutashap_gini", "borutashap_shap"}:
        import pandas as pd
        import numpy as np
        import scipy.stats

        # The original Boruta-Shap release imports scipy.stats.binom_test,
        # removed in modern SciPy. Restore its legacy p-value call locally in
        # this disposable benchmark process; do not patch package files.
        if not hasattr(scipy.stats, "binom_test"):
            scipy.stats.binom_test = lambda x, n=None, p=0.5, alternative="two-sided": scipy.stats.binomtest(
                x, n=n, p=p, alternative=alternative
            ).pvalue
        if not hasattr(np, "NaN"):
            np.NaN = np.nan
        import shap
        from BorutaShap import BorutaShap

        class CompatBorutaShap(BorutaShap):
            """Adapt modern SHAP tensor output to the legacy Boruta-Shap API."""

            def explain(self):
                explainer = shap.TreeExplainer(
                    self.model, feature_perturbation="tree_path_dependent", approximate=True
                )
                raw = explainer.shap_values(self.X_boruta)
                if isinstance(raw, list):
                    values = np.stack([np.asarray(item) for item in raw], axis=-1)
                else:
                    values = np.asarray(raw)
                if values.ndim == 3:
                    self.shap_values = np.abs(values).sum(axis=-1).mean(axis=0)
                else:
                    self.shap_values = np.abs(values).mean(axis=0)

        importance = "gini" if args.method.endswith("gini") else "shap"
        shap_model = RandomForestClassifier(
            n_estimators=args.trees, max_depth=16, max_features="sqrt",
            n_jobs=1, random_state=args.seed,
        )
        prime_seconds = 0.0
        if importance == "gini":
            # Original Boruta-Shap checks for fitted feature_importances_ in
            # __init__, then refits inside fit(). Include that required fit in
            # the end-to-end timing instead of hiding the overhead.
            prime_started = time.perf_counter()
            shap_model.fit(X, y)
            prime_seconds = time.perf_counter() - prime_started
        model = CompatBorutaShap(
            model=shap_model,
            importance_measure=importance,
            classification=True,
            percentile=100,
            pvalue=0.01,
        )
        setup_started = time.perf_counter()
        frame = pd.DataFrame(X, columns=names)
        setup_seconds = time.perf_counter() - setup_started
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        model.fit(
            X=frame, y=pd.Series(y), n_trials=args.iterations,
            random_state=args.seed, sample=False, train_or_test="train",
            normalize=False, verbose=False,
        )
        accepted = set(model.accepted)
        selected = np.asarray([i for i, name in enumerate(names) if name in accepted])
        n_iter = int(model.history_hits.shape[0] - 1)
        extra.update({
            "importance_method": importance,
            "accepted_count": len(model.accepted),
            "tentative_count": len(model.tentative),
            "gini_required_prime_fit_seconds": prime_seconds,
        })
    elif args.method == "python_orchestration_xrf":
        from bankai_random_forest import BankaiRandomForestClassifier

        rng = np.random.default_rng(args.seed)
        X_work = X.copy()
        hits = np.zeros(args.features, dtype=np.int64)
        rss_before = process.memory_info().rss
        started = time.perf_counter()
        for iteration in range(args.iterations):
            shadow = X_work.copy()
            for j in range(shadow.shape[1]):
                shadow[:, j] = shadow[rng.permutation(args.rows), j]
            augmented = np.concatenate((X_work, shadow), axis=1)
            forest = BankaiRandomForestClassifier(
                n_estimators=args.trees, max_depth=16, max_features="sqrt",
                n_jobs=1, random_state=args.seed + iteration,
                importance_type="permutation",
            ).fit(augmented, y)
            importance = forest.feature_importances_
            hits += importance[:args.features] > np.max(importance[args.features:])
        selected = np.flatnonzero(hits > args.iterations // 2)
        n_iter = args.iterations
        extra.update({
            "comparison_role": "Python-controlled repeated forest/shadow loop; diagnostic, not statistical Boruta",
            "iterations": args.iterations,
            "matrix_materialized_each_iteration": True,
            "copied_bytes_estimate": int(args.iterations * augmented.nbytes),
            "engine_revision": args.xrf_revision,
        })
    else:
        raise ValueError(f"unknown method {args.method}")

    elapsed = time.perf_counter() - started
    rss_after = process.memory_info().rss
    result = {
        "method": args.method,
        "scenario": args.scenario,
        "seed": args.seed,
        "rows": args.rows,
        "features": args.features,
        "trees": args.trees,
        "iteration_limit": args.iterations,
        "n_iter": n_iter,
        "fit_seconds": elapsed,
        "input_setup_seconds": setup_seconds,
        "end_to_end_seconds": elapsed + setup_seconds + extra.get("gini_required_prime_fit_seconds", 0.0),
        "rss_before_fit_bytes": rss_before,
        "rss_after_fit_bytes": rss_after,
        "rss_delta_bytes": rss_after - rss_before,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "python": platform.python_version(),
        **_score(selected, relevant, args.features),
        **extra,
    }
    print(json.dumps(result, sort_keys=True))


def _worker_command(args, method, scenario, seed, rows, features, trees=None, iterations=None):
    return [
        sys.executable, str(Path(__file__).resolve()), "--worker",
        "--method", method, "--scenario", scenario, "--seed", str(seed),
        "--rows", str(rows), "--features", str(features),
        "--trees", str(trees if trees is not None else args.trees),
        "--iterations", str(iterations if iterations is not None else args.iterations),
        "--xrf-revision", args.xrf_revision,
    ]


def _summarize(records):
    groups = {}
    for record in records:
        if record["phase"] == "warmup":
            continue
        key = (record["phase"], record["method"], record["scenario"], record["rows"], record["features"])
        groups.setdefault(key, []).append(record)
    out = []
    for (phase, method, scenario, rows, features), values in sorted(groups.items()):
        times = [item["fit_seconds"] for item in values]
        deltas = [item["rss_delta_bytes"] for item in values]
        include_quality = phase == "quality"
        out.append({
            "phase": phase, "method": method, "scenario": scenario, "rows": rows, "features": features,
            "runs": len(values), "median_fit_seconds": statistics.median(times),
            "median_end_to_end_seconds": statistics.median(item["end_to_end_seconds"] for item in values),
            "mean_fit_seconds": statistics.mean(times),
            "median_rss_delta_mib": statistics.median(deltas) / (1024 * 1024),
            "mean_precision": statistics.mean(item["precision"] for item in values) if include_quality else None,
            "mean_recall": statistics.mean(item["recall"] for item in values) if include_quality else None,
            "mean_false_positives": statistics.mean(item["false_positives"] for item in values) if include_quality else None,
            "mean_selected": statistics.mean(item["selected_count"] for item in values) if include_quality else None,
            "mean_n_iter": statistics.mean(item["n_iter"] for item in values if item["n_iter"] is not None)
            if any(item["n_iter"] is not None for item in values) else None,
        })
    return out


def _markdown(result):
    metadata = result["metadata"]
    lines = [
        "# M7 — Comparações limitadas a 10 mil linhas",
        "",
        f"Execução: {metadata['date']} · {metadata['platform']} · Python {metadata['python']}",
        f"sklearn {metadata['scikit_learn']} · NumPy {metadata['numpy']} · macOS arm64",
        "",
        "## Protocolo",
        "",
        f"- Todas as comparações, inclusive qualidade, usam exclusivamente {BENCHMARK_ROWS:,}×{BENCHMARK_FEATURES}; não há matriz multi-dataset.",
        f"- {metadata['repeats']} repetições por configuração de desempenho; casos de qualidade em 10k×100 com {metadata['quality_seeds']} seeds por cenário.",
        f"- Desempenho: {metadata['trees']} árvores e {metadata['iterations']} iterações. Qualidade exploratória: {metadata['quality_trees']} árvores, até {metadata['quality_iterations']} iterações e {metadata['quality_seeds']} seeds por cenário; profundidade máxima 16, um worker.",
        "- O orçamento de qualidade foi reduzido para três seeds e duas iterações por cenário após medir Boruta-Shap/SHAP em 10k; serve como checagem exploratória e não substitui o gate estatístico de 30 seeds.",
        "- O tempo de SHAP em 10k linhas motivou um orçamento de apenas duas iterações; esse resultado mede custo inicial, não convergência do seletor.",
        "- A fase de desempenho mede apenas tempo e memória; seleções após duas iterações não são métricas de qualidade e aparecem como traço no JSON bruto.",
        "- Tempo de fit exclui geração dos dados/imports. O total inclui criação de DataFrame quando exigida e o ajuste preliminar obrigatório de Boruta-Shap Gini.",
        "- RSS delta é a diferença de RSS do processo antes/depois do `fit`; não é pico nativo nem medida isolada de buffers transitórios. Valores negativos indicam que o RSS final ficou abaixo do inicial, não alocação negativa.",
        "- Boruta-Shap com Gini e SHAP aparece separado: SHAP muda o método de atribuição e não é uma comparação equivalente de importance.",
        "- `sklearn_rf` e `bankai_xrf_rf` são controles de floresta sem Boruta. `python_orchestration_xrf` mede um loop de shadow/importância em Python e não é Boruta estatístico.",
        "- `borutapy` e `borutapy_bankai_xrf` compartilham exatamente o controller, parâmetros, dados e importância Gini; essa é a comparação controlada do estimador sklearn contra o motor XRF Bankai.",
        "- `sklearn_rf` e `bankai_xrf_rf` escolhem top-2 somente para contextualizar o controle de floresta no cenário misto; essa seleção não é um resultado Boruta.",
        "- `boruta_fru` usa `fru-arrow` com o crate upstream `xrf` 0.1.7. `bankai_xrf_rf` e o diagnóstico Python usam o fork Bankai Random Forest no commit indicado.",
        "- `boruta_fru` mede controller + `fru-arrow` + XRF integrado. Não separa o custo do XRF upstream usando o mesmo controller Bankai, portanto não isola somente o efeito do motor.",
        "",
        "## Desempenho e seleção",
        "",
        "| Fase | Implementação | Cenário | Dados | Runs | Mediana total (s) | Mediana fit (s) | RSS delta mediano (MiB) | Precisão média | Recall médio | Falsos positivos | Features selecionadas | Iterações médias |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in result["summary"]:
        n_iter = "—" if row["mean_n_iter"] is None else f"{row['mean_n_iter']:.1f}"
        precision = "—" if row["mean_precision"] is None else f"{row['mean_precision']:.3f}"
        recall = "—" if row["mean_recall"] is None else f"{row['mean_recall']:.3f}"
        false_positives = "—" if row["mean_false_positives"] is None else f"{row['mean_false_positives']:.2f}"
        selected = "—" if row["mean_selected"] is None else f"{row['mean_selected']:.2f}"
        lines.append(
            f"| {row['phase']} | {row['method']} | {row['scenario']} | {row['rows']:,}×{row['features']} | {row['runs']} | "
            f"{row['median_end_to_end_seconds']:.4f} | {row['median_fit_seconds']:.4f} | {row['median_rss_delta_mib']:.2f} | "
            f"{precision} | {recall} | {false_positives} | {selected} | {n_iter} |"
        )
    lines += [
        "",
        "## Leitura e limites",
        "",
        "As métricas de precisão/recall usam os sinais sintéticos conhecidos e somente features confirmadas/aceitas. No cenário nulo, precisão/recall valem 1 quando nenhuma feature é selecionada; falsos positivos mostram a contagem diretamente. Implementações com políticas estatísticas distintas podem produzir seleções diferentes mesmo com seed/configuração nominal iguais. As medições são exploratórias deste host; a conclusão não é uma alegação universal de liderança.",
        "",
        "Os resultados completos por execução estão em [`m7_comparisons.json`](m7_comparisons.json). Testes, revisões e limitações de release estão em [`m7_validation_2026-09-27.json`](m7_validation_2026-09-27.json). O harness está em [`run_m7_comparisons.py`](../run_m7_comparisons.py).",
        "",
        "## Dependências e revisões",
        "",
        f"- BorutaPy: {metadata['borutapy_version']}; boruta-fru: {metadata['boruta_fru_version']}; Boruta-Shap: {metadata['borutashap_version']}",
        f"- pandas {metadata['pandas_version']}; Polars {metadata['polars_version']}; SciPy {metadata['scipy_version']}; SHAP {metadata['shap_version']}",
        "- Boruta-Shap original requer shims locais no processo descartável: `np.NaN`, `scipy.stats.binom_test` e redução da saída SHAP moderna (n×features×classes) a mean-absolute por feature. O harness preserva SHAP como atribuição, mas o patch de compatibilidade é explicitamente distinto do pacote upstream sem patch.",
        f"- bankai-boruta: {metadata['bankai_boruta_revision']}; bankai-random-forest: {metadata['xrf_revision']}",
        "- A execução é compatibilidade Python 3.12 para incluir boruta-fru; o projeto continua validado também em Python 3.11.",
        "- `boruta_fru` publica seus resultados com decisões por feature; o número de iterações efetivamente executadas não está exposto no API público e é omitido.",
        "- Referências: [boruta-fru](https://docs.rs/boruta-fru/0.1.0), [XRF 0.1.7](https://docs.rs/xrf/0.1.7), [BorutaPy](https://github.com/scikit-learn-contrib/boruta_py), [Boruta-Shap original](https://github.com/Ekeany/Boruta-Shap).",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--methods", nargs="+", choices=METHODS)
    parser.add_argument("--scenario", choices=("mixed", "null", "correlated", "xor"), default="mixed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=int, default=BENCHMARK_ROWS)
    parser.add_argument("--features", type=int, default=BENCHMARK_FEATURES)
    parser.add_argument("--trees", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--quality-trees", type=int, default=16)
    parser.add_argument("--quality-iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--xrf-revision", default="unknown")
    parser.add_argument("--bankai-revision", default="unknown")
    parser.add_argument("--quality-seeds", type=int, default=3)
    parser.add_argument("--quality-seed-offset", type=int, default=0,
                        help="start position in the stable 30-seed quality list")
    parser.add_argument("--quality-only", action="store_true",
                        help="run only the requested quality seed slice")
    parser.add_argument("--output-json", type=Path, default=ROOT / "benchmarks/results/m7_comparisons.json")
    parser.add_argument("--output-md", type=Path, default=ROOT / "benchmarks/results/m7_comparisons.md")
    args = parser.parse_args()
    if args.rows > 10_000:
        parser.error("M7 comparison protocol is capped at 10,000 rows")
    if min(args.trees, args.iterations, args.quality_trees, args.quality_iterations,
           args.repeats, args.quality_seeds) < 1:
        parser.error("tree counts, iteration limits, repeats, and quality-seeds must be positive")
    if args.quality_seed_offset < 0 or args.quality_seed_offset + args.quality_seeds > len(QUALITY_SEEDS):
        parser.error(f"quality seed slice must fit in the stable list of {len(QUALITY_SEEDS)} seeds")
    if args.worker:
        if not args.method:
            parser.error("--method is required in worker mode")
        worker(args)
        return

    methods = list(args.methods or METHODS)
    records = []

    def run_one(method, scenario, seed, rows, features, phase, trees=None, iterations=None):
        command = _worker_command(args, method, scenario, seed, rows, features, trees, iterations)
        result = json.loads(subprocess.check_output(command, cwd="/tmp", text=True))
        result["phase"] = phase
        records.append(result)
        print(f"{phase}: {method} {scenario} {rows}x{features} seed={seed} {result['fit_seconds']:.3f}s", flush=True)

    if not args.quality_only:
        # One untimed sample warms each path/dimension; records remain in raw JSON.
        for rows, features in PERFORMANCE_DIMENSIONS:
            warm_order = list(methods)
            random.Random(rows + 91).shuffle(warm_order)
            for method in warm_order:
                run_one(method, "mixed", 101, rows, features, "warmup")

        # Randomize implementation order per paired repeat to reduce drift.
        for rows, features in PERFORMANCE_DIMENSIONS:
            for repeat in range(args.repeats):
                order = list(methods)
                random.Random(10_000 * rows + repeat).shuffle(order)
                for method in order:
                    run_one(method, "mixed", 101 + repeat, rows, features, "performance")

    # Small but seeded quality cross-check on all synthetic scenarios.
    for method in (item for item in SELECTORS if item in methods):
        for scenario in ("mixed", "null", "correlated", "xor"):
            for seed in QUALITY_SEEDS[args.quality_seed_offset:args.quality_seed_offset + args.quality_seeds]:
                run_one(method, scenario, seed, BENCHMARK_ROWS, BENCHMARK_FEATURES, "quality",
                        trees=args.quality_trees, iterations=args.quality_iterations)

    import importlib.metadata as md

    metadata = {
        "date": time.strftime("%Y-%m-%d"),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": __import__("numpy").__version__,
        "scikit_learn": __import__("sklearn").__version__,
        "max_rows": 10_000,
        "methods": methods,
        "dimensions": [list(dimension) for dimension in PERFORMANCE_DIMENSIONS],
        "trees": args.trees,
        "iterations": args.iterations,
        "repeats": args.repeats,
        "quality_seeds": args.quality_seeds,
        "quality_seed_offset": args.quality_seed_offset,
        "quality_seed_values": list(QUALITY_SEEDS[args.quality_seed_offset:args.quality_seed_offset + args.quality_seeds]),
        "quality_trees": args.quality_trees,
        "quality_iterations": args.quality_iterations,
        "warmups_per_method_and_dimension": 1,
        "quality_dimensions": [BENCHMARK_ROWS, BENCHMARK_FEATURES],
        "comparison_order": "randomized deterministically per repeat; warmups precede measurements",
        "bankai_boruta_revision": args.bankai_revision,
        "xrf_revision": args.xrf_revision,
        "borutapy_version": md.version("Boruta"),
        "boruta_fru_version": md.version("boruta-fru"),
        "borutashap_version": md.version("BorutaShap"),
        "polars_version": md.version("polars"),
        "pandas_version": md.version("pandas"),
        "scipy_version": md.version("scipy"),
        "shap_version": md.version("shap"),
    }
    result = {"metadata": metadata, "summary": _summarize(records), "runs": records}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.write_text(_markdown(result), encoding="utf-8")
    print(f"wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
