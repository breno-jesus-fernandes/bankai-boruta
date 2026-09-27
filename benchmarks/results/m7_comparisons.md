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
- Tempo de fit exclui geração dos dados/imports. O total inclui criação de DataFrame quando exigida e o pré-fit necessário pelo construtor Boruta-Shap (Gini ou LightGBM).
- RSS delta é a diferença de RSS do processo antes/depois do `fit`; não é pico nativo nem medida isolada de buffers transitórios. Valores negativos indicam que o RSS final ficou abaixo do inicial, não alocação negativa.
- Boruta-Shap com Gini e SHAP aparece separado: SHAP muda o método de atribuição e não é uma comparação equivalente de importance.
- `sklearn_rf` e `bankai_xrf_rf` são controles de floresta sem Boruta. `python_orchestration_xrf` mede um loop de shadow/importância em Python e não é Boruta estatístico.
- `borutapy` e `borutapy_bankai_xrf` compartilham exatamente o controller, parâmetros, dados e importância Gini; essa é a comparação controlada do estimador sklearn contra o motor XRF Bankai.
- Par principal controlado: `borutapy_lightgbm` versus `borutashap_lightgbm_gain` usa o mesmo LightGBM, importância gain, dados, seed e orçamento; muda o controller Boruta. `borutashap_lightgbm_shap` mantém o LightGBM e muda explicitamente a atribuição para SHAP.
- O LightGBM pareado usa max_depth=16, num_leaves=31, feature fraction √p/p, class_weight=balanced, uma thread e modo determinístico. Os demais seletores ficam como baselines do ecossistema, com seus motores/atribuições próprios; não são tratados como comparação controlada de estimador.
- `sklearn_rf` e `bankai_xrf_rf` escolhem top-2 somente para contextualizar o controle de floresta no cenário misto; essa seleção não é um resultado Boruta.
- `boruta_fru` usa `fru-arrow` com o crate upstream `xrf` 0.1.7. `bankai_xrf_rf` e o diagnóstico Python usam o fork Bankai Random Forest no commit indicado.
- `boruta_fru` mede controller + `fru-arrow` + XRF integrado. Não separa o custo do XRF upstream usando o mesmo controller Bankai, portanto não isola somente o efeito do motor.

## Desempenho e checagem de seleção

| Fase | Implementação | Cenário | Dados | Runs | Mediana total (s) | Mediana fit (s) | RSS delta mediano (MiB) | Precisão média | Recall médio | Falsos positivos | Features selecionadas | Iterações médias |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| performance | bankai_boruta | mixed | 10,000×100 | 5 | 1.1707 | 1.1707 | 13.20 | — | — | — | — | 2.0 |
| performance | bankai_xrf_rf | mixed | 10,000×100 | 5 | 0.4406 | 0.4406 | 19.42 | — | — | — | — | — |
| performance | boruta_fru | mixed | 10,000×100 | 5 | 0.8302 | 0.8123 | 18.34 | — | — | — | — | — |
| performance | borutapy | mixed | 10,000×100 | 5 | 1.2138 | 1.2138 | 39.16 | — | — | — | — | 2.0 |
| performance | borutapy_bankai_xrf | mixed | 10,000×100 | 5 | 1.2868 | 1.2868 | 74.33 | — | — | — | — | 2.0 |
| performance | borutapy_lightgbm | mixed | 10,000×100 | 5 | 0.4908 | 0.4908 | 108.11 | — | — | — | — | 2.0 |
| performance | borutashap_gini | mixed | 10,000×100 | 5 | 1.7217 | 1.2795 | 54.58 | — | — | — | — | 2.0 |
| performance | borutashap_lightgbm_gain | mixed | 10,000×100 | 5 | 0.6386 | 0.5177 | 93.25 | — | — | — | — | 2.0 |
| performance | borutashap_lightgbm_shap | mixed | 10,000×100 | 5 | 0.7903 | 0.6536 | 109.62 | — | — | — | — | 2.0 |
| performance | borutashap_shap | mixed | 10,000×100 | 5 | 32.5970 | 32.5942 | -61.72 | — | — | — | — | 2.0 |
| performance | python_orchestration_xrf | mixed | 10,000×100 | 5 | 1.3151 | 1.3151 | 89.62 | — | — | — | — | 2.0 |
| performance | sklearn_rf | mixed | 10,000×100 | 5 | 0.4372 | 0.4372 | 6.11 | — | — | — | — | — |
| quality | bankai_boruta | correlated | 10,000×100 | 3 | 2.2028 | 2.2028 | 14.58 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | bankai_boruta | mixed | 10,000×100 | 3 | 2.3028 | 2.3028 | 13.56 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | bankai_boruta | null | 10,000×100 | 3 | 2.3522 | 2.3522 | 13.19 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | bankai_boruta | xor | 10,000×100 | 3 | 2.3219 | 2.3219 | 13.22 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | boruta_fru | correlated | 10,000×100 | 3 | 1.4943 | 1.4899 | 20.27 | 0.000 | 0.000 | 0.00 | 0.00 | — |
| quality | boruta_fru | mixed | 10,000×100 | 3 | 1.6007 | 1.5912 | 18.80 | 0.000 | 0.000 | 0.00 | 0.00 | — |
| quality | boruta_fru | null | 10,000×100 | 3 | 1.6537 | 1.6495 | 18.09 | 1.000 | 1.000 | 0.00 | 0.00 | — |
| quality | boruta_fru | xor | 10,000×100 | 3 | 1.6802 | 1.6756 | 17.78 | 0.000 | 0.000 | 0.00 | 0.00 | — |
| quality | borutapy | correlated | 10,000×100 | 3 | 2.4824 | 2.4824 | 39.62 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy | mixed | 10,000×100 | 3 | 2.4068 | 2.4068 | 40.06 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy | null | 10,000×100 | 3 | 2.5167 | 2.5167 | 40.16 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy | xor | 10,000×100 | 3 | 2.4137 | 2.4137 | 40.20 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_bankai_xrf | correlated | 10,000×100 | 3 | 2.2119 | 2.2119 | 76.36 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_bankai_xrf | mixed | 10,000×100 | 3 | 2.2381 | 2.2381 | 75.44 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_bankai_xrf | null | 10,000×100 | 3 | 2.2922 | 2.2922 | 75.95 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_bankai_xrf | xor | 10,000×100 | 3 | 2.2472 | 2.2472 | 76.20 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_lightgbm | correlated | 10,000×100 | 3 | 0.5069 | 0.5069 | 108.16 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_lightgbm | mixed | 10,000×100 | 3 | 0.5097 | 0.5097 | 108.12 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_lightgbm | null | 10,000×100 | 3 | 0.5141 | 0.5141 | 108.33 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutapy_lightgbm | xor | 10,000×100 | 3 | 0.5064 | 0.5064 | 107.67 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_gini | correlated | 10,000×100 | 3 | 3.3503 | 2.4824 | 55.25 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_gini | mixed | 10,000×100 | 3 | 3.3270 | 2.4812 | 58.78 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_gini | null | 10,000×100 | 3 | 3.3906 | 2.5152 | 58.86 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_gini | xor | 10,000×100 | 3 | 3.3580 | 2.5091 | 58.78 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_lightgbm_gain | correlated | 10,000×100 | 3 | 0.6836 | 0.5558 | 93.27 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_lightgbm_gain | mixed | 10,000×100 | 3 | 0.6852 | 0.5583 | 93.53 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_lightgbm_gain | null | 10,000×100 | 3 | 0.6796 | 0.5520 | 93.19 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_lightgbm_gain | xor | 10,000×100 | 3 | 0.6772 | 0.5495 | 91.31 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_lightgbm_shap | correlated | 10,000×100 | 3 | 0.9318 | 0.8050 | 110.41 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_lightgbm_shap | mixed | 10,000×100 | 3 | 0.9435 | 0.8154 | 110.11 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_lightgbm_shap | null | 10,000×100 | 3 | 0.9647 | 0.8379 | 110.36 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_lightgbm_shap | xor | 10,000×100 | 3 | 0.9581 | 0.8328 | 108.27 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_shap | correlated | 10,000×100 | 3 | 46.0938 | 46.0907 | 51.22 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_shap | mixed | 10,000×100 | 3 | 71.7638 | 71.7610 | -42.86 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_shap | null | 10,000×100 | 3 | 87.1645 | 87.1612 | -11.00 | 1.000 | 1.000 | 0.00 | 0.00 | 2.0 |
| quality | borutashap_shap | xor | 10,000×100 | 3 | 82.6032 | 82.5998 | -40.95 | 0.000 | 0.000 | 0.00 | 0.00 | 2.0 |

## Leitura e limites

As métricas de precisão/recall usam os sinais sintéticos conhecidos e somente features confirmadas/aceitas. Com duas iterações, todos os métodos deixaram os sinais como não selecionados; esta checagem não mede qualidade/convergência e serve somente para validar a execução. No cenário nulo, precisão/recall valem 1 quando nenhuma feature é selecionada; falsos positivos mostram a contagem diretamente. Implementações com políticas estatísticas distintas podem produzir seleções diferentes mesmo com seed/configuração nominal iguais. As medições são exploratórias deste host; a conclusão não é uma alegação universal de liderança.

Os resultados completos por execução estão em [`m7_comparisons.json`](m7_comparisons.json). Testes, revisões e limitações de release estão em [`m7_validation_2026-09-27.json`](m7_validation_2026-09-27.json). O harness está em [`run_m7_comparisons.py`](../run_m7_comparisons.py).

## Dependências e revisões

- BorutaPy: 0.4.3; boruta-fru: 0.1.0; Boruta-Shap: 1.0.17
- LightGBM: 4.7.0
- pandas 3.0.6; Polars 1.44.2; SciPy 1.18.1; SHAP 0.52.0
- Boruta-Shap original requer shims locais no processo descartável: `np.NaN`, `scipy.stats.binom_test` e redução da saída SHAP moderna (n×features×classes) a mean-absolute por feature. O harness preserva SHAP como atribuição, mas o patch de compatibilidade é explicitamente distinto do pacote upstream sem patch.
- bankai-boruta: 6ca70d3af749676cbcb6db1c45c7a6525f9e492f; bankai-random-forest: b29a7fc07fc36405a17c7739aaefef99baa72d79
- A execução é compatibilidade Python 3.12 para incluir boruta-fru; o projeto continua validado também em Python 3.11.
- `boruta_fru` publica seus resultados com decisões por feature; o número de iterações efetivamente executadas não está exposto no API público e é omitido.
- Referências: [LightGBM LGBMClassifier](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html), [boruta-fru](https://docs.rs/boruta-fru/0.1.0), [XRF 0.1.7](https://docs.rs/xrf/0.1.7), [BorutaPy](https://github.com/scikit-learn-contrib/boruta_py), [Boruta-Shap original](https://github.com/Ekeany/Boruta-Shap).
