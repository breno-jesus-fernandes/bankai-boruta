# M7 — Comparações limitadas a 10 mil linhas

Execução: 2026-09-27 · macOS-26.6.2-arm64-arm-64bit · Python 3.12.3
sklearn 1.9.1 · NumPy 2.5.3 · macOS arm64

## Protocolo

- Todas as comparações, inclusive qualidade, usam exclusivamente 10,000×100; não há matriz multi-dataset.
- 5 repetições por configuração de desempenho; casos de qualidade em 10k×100 com 3 seeds por cenário.
- Desempenho: 8 árvores e 2 iterações. Qualidade exploratória: 16 árvores, até 2 iterações e 3 seeds por cenário; profundidade máxima 16, um worker.
- O orçamento de qualidade foi reduzido para três seeds e duas iterações por cenário após medir Boruta-Shap/SHAP em 10k; serve como checagem exploratória e não substitui o gate estatístico de 30 seeds.
- O tempo de SHAP em 10k linhas motivou um orçamento de apenas duas iterações; esse resultado mede custo inicial, não convergência do seletor.
- A fase de desempenho mede apenas tempo e memória; seleções após duas iterações não são métricas de qualidade e aparecem como traço no JSON bruto.
- Tempo de fit exclui geração dos dados/imports. O total inclui criação de DataFrame quando exigida e o ajuste preliminar obrigatório de Boruta-Shap Gini.
- RSS delta é a diferença de RSS do processo antes/depois do `fit`; não é pico nativo nem medida isolada de buffers transitórios. Valores negativos indicam que o RSS final ficou abaixo do inicial, não alocação negativa.
- Boruta-Shap com Gini e SHAP aparece separado: SHAP muda o método de atribuição e não é uma comparação equivalente de importance.
- `sklearn_rf` e `bankai_xrf_rf` são controles de floresta sem Boruta. `python_orchestration_xrf` mede um loop de shadow/importância em Python e não é Boruta estatístico.
- `borutapy` e `borutapy_bankai_xrf` compartilham exatamente o controller, parâmetros, dados e importância Gini; essa é a comparação controlada do estimador sklearn contra o motor XRF Bankai.
- `sklearn_rf` e `bankai_xrf_rf` escolhem top-2 somente para contextualizar o controle de floresta no cenário misto; essa seleção não é um resultado Boruta.
- `boruta_fru` usa `fru-arrow` com o crate upstream `xrf` 0.1.7. `bankai_xrf_rf` e o diagnóstico Python usam o fork Bankai Random Forest no commit indicado.
- `boruta_fru` mede controller + `fru-arrow` + XRF integrado. Não separa o custo do XRF upstream usando o mesmo controller Bankai, portanto não isola somente o efeito do motor.

## Desempenho e seleção

| Fase | Implementação | Cenário | Dados | Runs | Mediana total (s) | Mediana fit (s) | RSS delta mediano (MiB) | Precisão média | Recall médio | Falsos positivos | Features selecionadas | Iterações médias |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| performance | bankai_boruta | mixed | 10,000×100 | 5 | 1.1590 | 1.1590 | 13.34 | — | — | — | — | 2.0 |
| performance | bankai_xrf_rf | mixed | 10,000×100 | 5 | 0.4460 | 0.4460 | 19.48 | — | — | — | — | — |
| performance | boruta_fru | mixed | 10,000×100 | 5 | 0.8019 | 0.7975 | 16.84 | — | — | — | — | — |
| performance | borutapy | mixed | 10,000×100 | 5 | 1.2184 | 1.2184 | 39.23 | — | — | — | — | 2.0 |
| performance | borutapy_bankai_xrf | mixed | 10,000×100 | 5 | 1.3030 | 1.3030 | 74.22 | — | — | — | — | 2.0 |
| performance | borutashap_gini | mixed | 10,000×100 | 5 | 1.7375 | 1.3024 | 54.58 | — | — | — | — | 2.0 |
| performance | borutashap_shap | mixed | 10,000×100 | 5 | 33.2356 | 33.2329 | 6.39 | — | — | — | — | 2.0 |
| performance | python_orchestration_xrf | mixed | 10,000×100 | 5 | 1.3351 | 1.3351 | 89.64 | — | — | — | — | 2.0 |
| performance | sklearn_rf | mixed | 10,000×100 | 5 | 0.4418 | 0.4418 | 6.14 | — | — | — | — | — |
| quality | bankai_boruta | correlated | 10,000×100 | 3 | 2.2676 | 2.2676 | 14.64 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | bankai_boruta | mixed | 10,000×100 | 3 | 2.3295 | 2.3295 | 13.55 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | bankai_boruta | null | 10,000×100 | 3 | 2.4027 | 2.4027 | 13.06 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | bankai_boruta | xor | 10,000×100 | 3 | 2.3451 | 2.3451 | 12.95 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | boruta_fru | correlated | 10,000×100 | 3 | 1.6319 | 1.6271 | 17.42 | 0.000 | 0.000 | 0.00 | 0.00 | — |
| quality | boruta_fru | mixed | 10,000×100 | 3 | 1.7231 | 1.7185 | 17.89 | 0.000 | 0.000 | 0.00 | 0.00 | — |
| quality | boruta_fru | null | 10,000×100 | 3 | 1.8283 | 1.8235 | 17.72 | 1.000 | 1.000 | 0.00 | 0.00 | — |
| quality | boruta_fru | xor | 10,000×100 | 3 | 1.7316 | 1.7267 | 13.55 | 0.000 | 0.000 | 0.00 | 0.00 | — |
| quality | borutapy | correlated | 10,000×100 | 3 | 2.6222 | 2.6222 | 39.55 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy | mixed | 10,000×100 | 3 | 2.4519 | 2.4519 | 40.12 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy | null | 10,000×100 | 3 | 2.5407 | 2.5407 | 40.12 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy | xor | 10,000×100 | 3 | 2.5203 | 2.5203 | 40.31 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_bankai_xrf | correlated | 10,000×100 | 3 | 2.4896 | 2.4896 | 76.39 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_bankai_xrf | mixed | 10,000×100 | 3 | 2.5372 | 2.5372 | 75.81 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_bankai_xrf | null | 10,000×100 | 3 | 2.5577 | 2.5577 | 75.95 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_bankai_xrf | xor | 10,000×100 | 3 | 2.5141 | 2.5141 | 75.89 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_gini | correlated | 10,000×100 | 3 | 3.3795 | 2.5300 | 54.64 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_gini | mixed | 10,000×100 | 3 | 3.5787 | 2.6089 | 24.19 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_gini | null | 10,000×100 | 3 | 3.4094 | 2.4890 | 58.86 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_gini | xor | 10,000×100 | 3 | 3.3252 | 2.4803 | 58.86 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_shap | correlated | 10,000×100 | 3 | 46.1353 | 46.1309 | -88.05 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_shap | mixed | 10,000×100 | 3 | 71.0872 | 71.0845 | 10.86 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_shap | null | 10,000×100 | 3 | 86.4462 | 86.4424 | -100.94 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_shap | xor | 10,000×100 | 3 | 81.7949 | 81.7917 | -54.84 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |

## Leitura e limites

As métricas de precisão/recall usam os sinais sintéticos conhecidos e somente features confirmadas/aceitas. No cenário nulo, precisão/recall valem 1 quando nenhuma feature é selecionada; falsos positivos mostram a contagem diretamente. Implementações com políticas estatísticas distintas podem produzir seleções diferentes mesmo com seed/configuração nominal iguais. As medições são exploratórias deste host; a conclusão não é uma alegação universal de liderança.

Os resultados completos por execução estão em [`m7_comparisons.json`](m7_comparisons.json). Testes, revisões e limitações de release estão em [`m7_validation_2026-09-27.json`](m7_validation_2026-09-27.json). O harness está em [`run_m7_comparisons.py`](../run_m7_comparisons.py).

## Dependências e revisões

- BorutaPy: 0.4.3; boruta-fru: 0.1.0; Boruta-Shap: 1.0.17
- pandas 3.0.6; Polars 1.44.2; SciPy 1.18.1; SHAP 0.52.0
- Boruta-Shap original requer shims locais no processo descartável: `np.NaN`, `scipy.stats.binom_test` e redução da saída SHAP moderna (n×features×classes) a mean-absolute por feature. O harness preserva SHAP como atribuição, mas o patch de compatibilidade é explicitamente distinto do pacote upstream sem patch.
- bankai-boruta: 6ca70d3af749676cbcb6db1c45c7a6525f9e492f; bankai-random-forest: b29a7fc07fc36405a17c7739aaefef99baa72d79
- A execução é compatibilidade Python 3.12 para incluir boruta-fru; o projeto continua validado também em Python 3.11.
- `boruta_fru` publica seus resultados com decisões por feature; o número de iterações efetivamente executadas não está exposto no API público e é omitido.
- Referências: [boruta-fru](https://docs.rs/boruta-fru/0.1.0), [XRF 0.1.7](https://docs.rs/xrf/0.1.7), [BorutaPy](https://github.com/scikit-learn-contrib/boruta_py), [Boruta-Shap original](https://github.com/Ekeany/Boruta-Shap).
