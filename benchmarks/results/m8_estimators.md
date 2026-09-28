# M8 — Benchmarks de estimadores externos (10k×100)

Execução: 2026-09-27 · macOS-26.6.2-arm64-arm-64bit · Python 3.11.11
scikit-learn 1.9.1 · NumPy 2.4.6

## Protocolo

- Todas as execuções usam somente 10.000×100, dados sintéticos mistos e processos isolados.
- 5 repetições medidas após um aquecimento por método; 8 árvores, até 2 iterações, profundidade 16 e uma thread.
- `bankai_external_sklearn` versus `borutapy_sklearn` é o par justo: ambos usam o mesmo RandomForest sklearn, importância impurity, seed e configuração. Muda o controller e a materialização do adapter.
- `bankai_native` usa o provider Rust/XRF e importância permutation; compara o caminho default completo, não isola somente o overhead do adapter.
- `bankai_external_xrf` usa a importância permutation do BankaiRandomForestClassifier. O getter subtrai o mínimo global do vetor por iteração, preservando a ordem real-vs-shadow e tornando os scores não negativos para o contrato do selector.
- Com duas iterações, os resultados não permitem concluir qualidade, convergência ou liderança universal.
- RSS é delta do processo antes/depois do fit, não pico de memória. Bytes copiados contam a matriz aumentada conhecida; não incluem alocações internas do estimator.

## Resultados

| Método | Runs | Mediana fit (s) | RSS delta mediano (MiB) | Bytes X copiados | Callbacks medianos | Tempo callback (s) | Selecionadas | Iterações |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bankai_external_sklearn | 5 | 1.2039 | 34.38 | 32000000 | 2.0 | 1.2016 | 0.0 | 2.0 |
| bankai_external_xrf | 5 | 1.2936 | 60.38 | 32000000 | 2.0 | 1.2909 | 0.0 | 2.0 |
| bankai_native | 5 | 1.1070 | 13.23 | 0 | 0.0 | 0.0000 | 0.0 | 2.0 |
| borutapy_sklearn | 5 | 1.1946 | 39.05 | 0 | 0.0 | 0.0000 | 0.0 | 2.0 |

Os resultados por repetição, configurações e revisões estão em [`m8_estimators.json`](m8_estimators.json).
