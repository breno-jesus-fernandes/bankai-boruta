# bankai-boruta — Roadmap de desenvolvimento

Plano arquitetural e tático aprovado em 2026-09-26. A v1 será um seletor de
features para classificação binária e multiclasse, com loop inteiramente Rust,
integração NumPy/Polars e API de transformação compatível com scikit-learn.

## Princípios e decisões

- Uma chamada Python a `BankaiBoruta.fit(X, y)` executa geração de shadows,
  treinamento, importância, inferência e poda dentro do Rust.
- Extrair `bankai-forest-core`, sem PyO3, do bankai-random-forest-v2. Compartilhar
  o motor diretamente entre crates, sem chamar seu estimador Python por iteração.
- Fixar a dependência por commit inicialmente e por versão quando publicada.
  Não distribuir dependências com caminhos absolutos da máquina do autor.
- Permutation importance OOB nativa por padrão, com escala por média/desvio
  entre árvores (`zscore`); oferecer `raw` explicitamente.
- Modo `canonical` padrão: rejeitadas saem do treino; confirmadas permanecem
  como contexto e fontes de shadows, mas deixam de receber decisões.
- Modo `aggressive` experimental: confirmadas e rejeitadas saem do treino;
  confirmadas continuam selecionadas no resultado.
- Teste binomial e Bonferroni de referência. Não prometer controle global de
  erro para testes sequenciais repetidos.
- Zero-copy explícito para entradas compatíveis, com modo alternativo de uma
  cópia inicial. Nenhuma cópia integral da matriz a cada iteração.
- TDD estrito: testes vermelhos, implementação mínima, refatoração e medição.
- Maturin é o único backend de build/empacotamento Python. Cargo executa testes
  e verificações Rust; não constitui um backend Python alternativo.
- Preservar GPL-3.0-or-later e atribuições existentes no motor compartilhado.
- SHAP nativo, regressão, categóricos, dados esparsos, multioutput e histogramas
  ficam fora da v1. Os contratos devem permitir extensões futuras.

Velocidade será demonstrada por benchmarks. A poda reduz trabalho conforme o
conjunto ativo diminui; não há garantia de aceleração exponencial nem de ser a
implementação mais rápida em todos os cenários.

## Arquitetura e interface

```text
BankaiBoruta.fit(X, y)
  → validação e aquisição de buffers
  → boruta-core
      → visão de features reais e shadows
      → bankai-forest-core / fork XRF
      → importância OOB
      → testes estatísticos e atualização dos conjuntos ativos
  → resultados compactos para Python
```

- `bankai-forest-core`: armazenamento, treinamento e importância, extraídos
  do motor existente e independentes de Python.
- `boruta-core`: estados, shadows, política estatística e orquestração.
- `bankai_boruta._core`: fronteira PyO3, ownership e conversão de resultados.
- Trait Rust para fontes de importância: scores associados a IDs de features,
  com declaração da necessidade de reter árvores. Sem callbacks ou estimadores
  Python dentro do loop.

Interface inicial:

```python
BankaiBoruta(
    n_estimators=500,
    max_iter=100,
    alpha=0.01,
    pruning="canonical",
    importance="permutation",
    importance_scale="zscore",
    input_mode="zero_copy",
    max_features="sqrt",
    max_depth=512,
    n_jobs=1,
    random_state=None,
    store_history=False,
)
```

Métodos: `fit`, `transform`, `fit_transform`, `get_support` e
`get_feature_names_out`. Integrar `BaseEstimator`, `SelectorMixin`, `clone`,
`Pipeline` e busca de hiperparâmetros.

Resultados: `support_` (confirmadas), `support_weak_` (tentativas), `decision_`,
`n_features_in_`, `feature_names_in_` quando aplicável, `n_iter_`, `stop_reason_`
e `diagnostics_`. Este último registra cópias de entrada, bytes auxiliares,
tempos por etapa e contagens de features. Não resolver tentativas por heurística
automaticamente. Serializar parâmetros e resultados, sem dados de treino.

## Contrato de memória

- NumPy zero-copy: `float32`/`float64`, alinhamento válido e ordem C ou F.
  Layouts incompatíveis geram erro orientando o uso de `input_mode="copy"`.
- Polars: Arrow PyCapsule, preservando buffers, offsets e chunks; sem conversão
  intermediária para NumPy ou rechunk obrigatório.
- V1 aceita features numéricas finitas; recusa nulos, NaN, infinitos, categóricos
  e esparsos explicitamente.
- Codificar `y` uma única vez pode exigir `O(n)` memória. Zero-copy da matriz
  não significa ausência de alocações auxiliares.
- Preservar owners durante todo o `fit`; nenhuma referência emprestada sobrevive
  ao retorno. Proibir mutação externa durante o empréstimo.
- `PyReadonlyArray` não prova exclusividade de todos os aliases. No caminho
  inicial NumPy emprestado, manter o GIL; paralelismo Rust continua disponível.
  Liberar o GIL para armazenamento próprio ou imutabilidade garantida. O GIL
  sozinho não impede escrita por código nativo externo já em execução.
- Shadows virtuais acessam `X[permutação[row], coluna]`; usar permutações
  independentes por shadow e iteração. Não materializar `[X | shadows]`.
- Poda altera mapas de colunas e reutiliza buffers; a matriz original permanece
  alocada pelo chamador. Não prometer liberação física das colunas emprestadas.
- Compartilhar rótulos, descartar árvores após agregar a importância e manter
  histórico completo desligado por padrão.
- Para `s` shadows, permutações exigem `O(ns)` índices. Usar `u32` quando couber,
  com alternativa `usize` para entradas maiores. Estados estatísticos são
  `O(p)`; workspaces de árvores crescem com o número de workers.
- `transform` pode alocar sua saída; essa alocação não viola o contrato do `fit`.

## Protocolo de conclusão de marcos

Cada marco deve ser concluído antes de iniciar o seguinte. Dividir os maiores
em PRs/commits de apoio quando necessário, mantendo o checkpoint final abaixo.

1. Escrever e executar os testes correspondentes antes da implementação.
2. Implementar, refatorar e executar verificações pertinentes.
3. Atualizar este documento com status, evidências e limitações.
4. Revisar o diff e adicionar somente alterações pertencentes ao marco.
5. Fazer o commit de conclusão e push para a branch de trabalho no GitHub.
6. Confirmar a presença do commit no remoto antes de avançar.

No bankai-boruta, a branch inicial é `main` e o remoto é `origin`, em
<https://github.com/breno-jesus-fernandes/bankai-boruta>.
Não usar force-push. Se houver divergência ou falha de push, resolver sem
descartar alterações remotas e registrar impedimentos materiais.

Para mudanças no motor compartilhado, verificar sua branch e remoto antes de
publicar. Commitar e enviar a alteração do motor antes de fixar sua revisão no
Boruta. A autorização do mantenedor inclui commit e push ao final dos marcos.

## M0 — Publicar o roadmap

**Status:** concluído nesta entrega documental; commit/push são o checkpoint
de publicação deste próprio marco.

**Objetivo:** registrar contratos, marcos, riscos e critérios de pronto.

**Validação primeiro:** revisar Markdown, decisões, checklists e consistência
com o código inspecionado. Não requer testes de código.

**Memória:** distinguir zero-copy de entrada de alocações auxiliares.

**Risco:** apresentar metas de desempenho como resultados demonstrados.

- [x] Registrar arquitetura e contratos.
- [x] Separar objetivos, testes, memória e riscos por marco.
- [x] Definir benchmarks e rotina de commit/push.

**Commit:** `docs: add milestone-based development roadmap`

## M1 — Contratos executáveis e infraestrutura TDD

**Status:** concluído em 2026-09-27. O binding e o estimador já existem; o loop
nativo permanece deliberadamente indisponível até M4.

**Objetivo:** estabelecer comportamento e medição antes do motor funcional.

**Rust primeiro:** estados `Tentative`, `Confirmed`, `Rejected`; decisões
irreversíveis; IDs originais estáveis; limite de iterações; fornecedor artificial
de importâncias para testar a orquestração sem floresta.

**Python primeiro:** construtor, parâmetros clonáveis, importação da extensão,
`fit` retornando `self`, resultados e erro ao transformar antes de ajustar.

**Memória:** instrumentar bytes copiados e alocações; preparar pico RSS em
subprocessos. Estado estatístico sem histórico deve ser `O(p)`.

**Risco:** confundir equivalência estatística com igualdade entre florestas
distintas ou perpetuar detalhes acidentais de outra implementação.

- [x] Ledger Rust mantém IDs originais, estados tentativos e decisões finais
  irreversíveis; IDs fora do intervalo e tentativas de reset geram erro.
- [x] Binding PyO3 carregável e estimador sklearn clonável, com validação de
  parâmetros e erro claro de `fit` até a chegada do motor M4.
- [x] Fixtures seeded para sinal misto, nulo, correlação e XOR; baseline sklearn
  em JSON registra dataset, seed, versão, plataforma e duração.
- [x] `cargo fmt --all -- --check`, `cargo test --workspace` (6 testes),
  `cargo clippy --workspace --all-targets -- -D warnings`, `uv run pytest`
  (22 testes), baseline JSON e `git diff --check` aprovados.
- [x] Roadmap atualizado, commitado e enviado ao GitHub.

**Limitação registrada:** `BankaiBoruta.fit` lança `RuntimeError` até M4. O
ledger não escolhe decisões estatísticas ainda; isso será coberto pelo TDD de M4.

**Commit:** `test: establish boruta contracts and benchmark harness`

## M2 — Núcleo Rust compartilhado

**Status:** concluído em 2026-09-27. O crate `bankai-forest-core` foi extraído
e enviado no commit `6b0649f` à branch `codex/shared-core-extraction` do
repositório bankai-random-forest; M3 atualizou o pin para `ab90aa7` após
compartilhar o splitter exato.

**Objetivo:** extrair o motor sem PyO3 e sem duplicar o algoritmo da floresta.

**Rust primeiro:** paridade antes/depois da extração; importância com árvores
retidas e descartadas; agregadores vazios; features nunca usadas; OOB vazio;
compartilhamento dos rótulos sem cópia por árvore.

**Python primeiro:** suíte existente do bankai-random-forest e importação
simultânea das duas extensões, preservando a API do motor.

**Memória:** generalizar armazenamento emprestado/próprio; compartilhar rótulos
por referência ou `Arc`; reutilizar workspaces. Validar denominadores e merges
no caminho de descarte de árvores já existente.

**Riscos:** regressões de treino/importância, lifetimes e limites `Send`/`Sync`,
além de acoplamento entre publicações dos dois projetos.

- [x] `bankai-forest-core` contém `DenseInput`, tipos de importância e XRF
  reexportado, sem dependência de Python; o binding existente reexporta
  `dense` para manter os caminhos internos da API.
- [x] O agregador de permutation importance compartilha os rótulos por `Arc`;
  teste verifica identidade de alocação em vez de confiar apenas em RSS.
- [x] Testes do crate provam validação, treino de floresta sem Python e
  importância OOB; a suíte Python do motor passou (147 testes, 2 avisos
  numéricos já esperados em `log(0)`).
- [x] Boruta fixa a revisão Git `6b0649f4` e testa treino/importância do crate
  através da fronteira real de dependência entre repositórios.
- [x] `cargo test --workspace`, formatação dos arquivos novos e Clippy do core
  passaram. Clippy precisou suprimir três lints preexistentes no fork XRF
  (`assign_op_pattern`, `too_many_arguments`, `needless_range_loop`).
- [x] Alteração do motor commitada e enviada; roadmap Boruta atualizado.

**Limitação transferida para M3:** `DenseInput::training` ainda aceita
`Vec<f64>` próprio. A extração torna o motor reutilizável, mas não implementa
zero-copy de NumPy/Arrow nem reduz a cópia inicial da matriz.

- [x] Commit e push do checkpoint Boruta realizados.

**Commit:** `refactor: extract shared native forest core`

## M3 — NumPy zero-copy e shadows virtuais

**Status:** concluído em 2026-09-27. O adaptador usa o splitter exato compartilhado
(`bankai-forest-core` `ab90aa7`) e replica o voto binário com limites do core.

**Objetivo:** executar uma iteração completa sem materializar a matriz aumentada.

**Rust primeiro:** equivalência com matriz materializada de referência;
preservação do multiconjunto de cada coluna; permutações independentes; acesso
após poda; composição da permutação shadow com a permutação da importância.

**Python primeiro:** C/F, `float32`/`float64`, rejeição de layouts incompatíveis,
dimensões inválidas, entrada inalterada, owners vivos e paridade copy/zero-copy.

**Memória:** reutilizar a capacidade dos índices de permutação e preservar `X`;
instrumentar zero bytes copiados da matriz no caminho estrito.

**Riscos:** indireções prejudicarem cache, permutações compartilhadas preservarem
correlações indevidas e mutação externa invalidar o empréstimo.

- [x] Paridade de árvore e importância com `DenseInput` em 8 seeds, classes
  binárias/multiclasse e matrizes com valores repetidos.
- [x] Zero cópias integrais de `X` comprovadas pelo empréstimo de buffer e teste
  do ingresso NumPy; sombras são colunas virtuais.
- [x] Registrar custo auxiliar: `O(n * shadows)` índices por permutação e
  workspaces de treinamento por árvore.
- [x] Verificações concluídas, roadmap atualizado, commit e push realizados.

**Commit:** `feat: add zero-copy numpy input and virtual shadows`

Commit de conclusão: `1466e6f`.

## M4 — Boruta canônico completo

**Status:** concluído em 2026-09-27.
`BankaiBoruta.fit` faz uma única chamada PyO3 e executa todas as iterações em
Rust, com permutation importance OOB do core compartilhado.

**Objetivo:** implementar o loop nativo com inferência estatística de referência.

Política fixa:

- Hit somente quando a importância real supera estritamente a maior shadow.
- `zscore` é média/desvio entre árvores, não padronização entre features.
- Features ausentes no agregador recebem zero. Escala indefinida recebe zero
  e contador diagnóstico.
- Teste binomial com `p=0.5`: cauda superior inclusiva para confirmar e inferior
  inclusiva para rejeitar. Calcular caudas de modo numericamente estável.
- Bonferroni usa o número original de features, como no código R. Só decidir
  features ainda tentativas; comparar p-valor ajustado estritamente com `alpha`.
- Duplicar logicamente fontes shadow até haver ao menos cinco; permutar cada
  cópia independentemente.
- Remover rejeitadas e preservar confirmadas como contexto e fonte de shadows.
- Encerrar quando não houver tentativas ou após exatamente `max_iter` execuções.
- Bootstrap obrigatório para importância OOB. Árvores sem OOB não contribuem
  para seu denominador válido; uma iteração sem evidência OOB falha claramente.

**Rust primeiro:** caudas binomiais contra fixtures de alta precisão; extremos,
underflow, empates, normalização, uma feature, decisões simultâneas, IDs e parada.

**Python primeiro:** integração completa, sinal forte/ruído, determinismo com
seed fixa, tentativas legítimas no limite e uma única entrada no núcleo por fit.

**Memória:** estados `O(p)`, árvores descartadas e buffers reutilizados;
histórico opcional `O(Tp)`, sem matrizes SHAP.

**Riscos:** normalização incorreta, OOB insuficiente e interpretação indevida de
testes repetidos como garantia sequencial global.

- [x] Caudas binomiais inclusivas e Bonferroni contra IDs originais, com
  comparação estrita e decisão de hits pré-computada por iteração.
- [x] Testes de referência cobrem frações exatas, underflow, limites de alpha,
  multiclasse, decisões simultâneas, empates e parada exata em `max_iter`.
- [x] Seleção funcional: sombras virtuais, score zscore/raw, OOB, poda de
  rejeitadas, confirmadas preservadas como contexto e tentativas não resolvidas.
- [x] Testes Python cobrem sinais fortes/nulos, uma feature, rótulos string e
  multiclasse, determinismo seeded, C/F, f32/f64, cópia explícita e uma chamada
  nativa total.
- [x] `cargo fmt --all -- --check`, `cargo test --workspace` (30 testes),
  `cargo clippy --workspace --all-targets -- -D warnings`, build Maturin e
  `uv run pytest` (49 testes) aprovados.
- [x] Roadmap atualizado, commit e push realizados.

**Limitações observadas no checkpoint M4:** poda agressiva e execução paralela
ainda não estavam implementadas. O caminho zero-copy exige NumPy
contíguo/alinhado; Polars e outros formatos entram em M6.

**Commit:** `feat: implement canonical native boruta selection`

## M5 — Poda agressiva e paralelismo determinístico

**Status:** concluído em 2026-09-27; poda agressiva continua experimental.

**Objetivo:** reduzir trabalho removendo todas as features decididas no modo
agressivo e limitar overhead e memória por worker.

**Rust primeiro:** confirmadas continuam selecionadas após sair do treino;
features removidas não voltam ao sampler; `max_features` acompanha a dimensão
aumentada atual; sementes derivadas de iteração, ID original, shadow e árvore;
agregação em ordem fixa e reproduzível.

**Python primeiro:** comparação pareada entre modos para correlações, duplicatas,
sinais fracos e XOR; repetibilidade serial/paralela e resultados serializados.

**Memória:** compactar mapas e reutilizar buffers; workspaces por worker e filas
limitadas; não usar paralelismo aninhado.

**Riscos:** remoção de confirmadas destruir interações restantes, rejeições
precoces irreversíveis e multiplicação do consumo por threads.

- [x] Invariantes de poda e reprodutibilidade aprovadas; em 24 comparações
  pareadas serial/paralelo, as decisões foram idênticas.
- [x] Tempo, precisão e recall publicados em
  [`benchmarks/results/m5_pruning_parallel_2026-09-27.json`](../benchmarks/results/m5_pruning_parallel_2026-09-27.json):
  quatro cenários, três seeds, 12 fits por modo serial e 24 verificações de
  paridade. A média serial foi 10,21 s canônico e 10,92 s agressivo; os modos
  tiveram precisão média 0,917 e recall 0,875, com os erros concentrados no
  cenário XOR. Quatro workers aceleraram os pares em aproximadamente 3,5x.
- [x] Poda agressiva permanece experimental: neste protocolo ela não mostrou
  aceleração serial e o conjunto de três seeds não satisfaz o critério de
  confiança de qualidade definido abaixo.
- [x] `cargo fmt --all -- --check`, `git diff --check`, `cargo test --workspace`,
  `cargo clippy --workspace --all-targets -- -D warnings`, build Maturin e
  `uv run pytest` (51 testes) aprovados. Roadmap atualizado, commit e push.

**Commit:** `perf: add aggressive pruning and deterministic parallelism`

## M6 — Polars, sklearn e ciclo de vida

**Status:** concluído em 2026-09-27. NumPy e Polars entram no mesmo loop
nativo; Polars usa Arrow C Stream com buffers compartilhados e ownership RAII.

**Plano de conclusão:** [M6.1–M6.6](limitations/M6-resolution-plan.md), com
contrato de tipos, abstração colunar, ownership, TDD e gates de aceite.

**Objetivo:** fechar interoperabilidade, pipelines e serialização compacta.

**Rust primeiro:** Arrow com offsets/chunks, tipos numéricos suportados,
liberação exatamente uma vez, erros sem vazamentos e ausência de cópias dos
buffers importados.

**Python primeiro:** paridade NumPy/Polars, nomes e ordem, `clone`, `Pipeline`,
validação cruzada, métodos de seleção/transformação, novo fit limpando estado
anterior e pickle/joblib sem reter dados de treinamento.

**Memória:** importar Arrow sem passar por NumPy; não serializar owners, `X`
ou florestas. Permitir a alocação da saída selecionada de `transform`.

**Riscos:** conversões ocultas no exportador ou na validação sklearn e vazamento
de informação ao selecionar features antes da validação cruzada.

- [x] NumPy obrigatório e Polars opcional; zero-copy Arrow suporta Float32/64,
  chunks, slices/offsets e colunas de tipos mistos sem conversão NumPy.
- [x] Testes cobrem seleção em `Pipeline`, `clone`, nomes, refit removendo
  nomes antigos e pickle sem reter `X`, `y` ou floresta.
- [x] Ingresso Arrow, liberação em sucesso/erro parcial, pipelines, lifecycle,
  artefatos e contrato documentados em
  [`docs/limitations/M6-arrow-zero-copy.md`](limitations/M6-arrow-zero-copy.md).
- [x] Pickle round-trip do estimador treinado sem matrizes/árvore no estado; fit
  e import também passam após instalação isolada do wheel e do sdist.
- [x] Rust (40 testes), Python (76 testes), Clippy estrito, Miri no crate FFI,
  Polars 1.32, smoke wheel/sdist e benchmarks aprovados; evidências em
  [`m6_validation_2026-09-27.json`](../benchmarks/results/m6_validation_2026-09-27.json).
- [x] Benchmark NumPy pré/pós-refatoração: razão geométrica 1,0195x (1,95%
  mais lento, abaixo do limite de investigação de 10%); RSS de processo inclui
  fixtures e não representa alocação incremental por fit.

**Commit:** `feat: add polars and sklearn selector integration`

## M7 — Benchmarks, endurecimento e distribuição

**Status:** em andamento. As comparações foram reduzidas a um único dataset
10k×100. O conjunto completo de desempenho e a checagem exploratória de
qualidade foram executados. Falta observar CI em
plataformas não macOS; o Clippy estrito do XRF no projeto irmão segue com lints
preexistentes registrados no relatório de validação.

**Objetivo:** demonstrar desempenho e memória em instalações reproduzíveis.

**Rust primeiro:** regressões de alocação, fits repetidos sem crescimento retido,
overflow de índices/dimensões e segurança dos adaptadores com Miri/sanitizers
onde aplicável.

**Python primeiro:** wheel e sdist instalados em ambientes limpos, smoke tests
sem checkout do motor irmão, versões Python suportadas e erros sem aborto do
processo.

**Memória:** medir RSS total, incremento durante fit, buffers auxiliares e bytes
copiados. `tracemalloc` sozinho não mede as alocações Rust.

**Riscos:** diferenças de importância/stopping invalidarem comparações,
oversubscription e resultados dominados por conversão ou compilação.

- [x] Comparações executadas exclusivamente em 10k×100: 180 resultados finais,
  incluindo 5 repetições de desempenho, warmups e qualidade exploratória com
  3 seeds e 2 iterações em quatro cenários para nove seletores. O par BorutaPy
  vs Boruta-Shap com LightGBM gain usa configuração idêntica de estimador; a
  variante SHAP é reportada separadamente. A checagem de qualidade não atende
  ao gate de 30 seeds e não mede convergência neste orçamento. Suítes locais
  aprovadas: Boruta 85 Python (3.11 e 3.13),
  40 Rust; Random Forest 147 Python e 20 Rust. O Clippy estrito do projeto irmão
  permanece com lints anteriores ao benchmark. Evidências em
  [`m7_validation_2026-09-27.json`](../benchmarks/results/m7_validation_2026-09-27.json).
  Resultados e limitações: [`m7_comparisons.md`](../benchmarks/results/m7_comparisons.md)
  e [`m7_comparisons.json`](../benchmarks/results/m7_comparisons.json).
- [x] Distribuições locais instaláveis e smoke test isolado no macOS arm64,
  Python 3.11; evidências e hashes em
  [`benchmarks/results/m7_artifact_smoke_2026-09-27.json`](../benchmarks/results/m7_artifact_smoke_2026-09-27.json).
- [x] Suítes locais dos dois projetos aprovadas; execução CI em plataformas não
  macOS ainda pendente.
- [x] Limitações e perdas de desempenho registradas; a medição de SHAP em
  10k×100 usou somente duas iterações e não é evidência de convergência nem
  declaração de liderança.
- [x] Wheel e sdist locais construídos exclusivamente com Maturin e instalados
  em ambientes isolados, conforme o relatório de smoke do M7.
- [x] Roadmap atualizado, commit e push realizados.

**Commit:** `release: validate benchmarks and distribution artifacts`

Publicar código e relatórios no GitHub faz parte do marco. Publicação em um
registro de pacotes não é pré-requisito e exige configuração própria de release.

## Benchmarks e definição de pronto

### Comparações

| Comparação | Finalidade |
| --- | --- |
| Controlador Boruta de referência + sklearn RandomForest | Baseline completo |
| XRF upstream fixado + mesmo controlador Boruta | Isolar ganhos do fork e integração |
| Bankai chamado por iteração em Python | Quantificar FFI e cópias |
| boruta-fru | Concorrente nativo direto |
| BorutaPy e Boruta-Shap | Comparações adicionais do ecossistema |

Para esta rodada, a pedido do mantenedor, usar exclusivamente 10k×100. Os
datasets de 1k, 100k e alta dimensionalidade ficam fora desta matriz. O relatório distingue
as métricas de floresta isolada, BorutaPy, Boruta-fru, Boruta-Shap com Gini/SHAP,
BorutaPy e Boruta-Shap pareados com LightGBM, o seletor Bankai e um diagnóstico
de orquestração Python. Não tratar
`sklearn.inspection.permutation_importance` em dados distintos como equivalente
à importância OOB; explicitar diferenças de importância e controller. A amostra
de qualidade com três seeds e duas iterações é preliminar e não atende ao gate
de trinta seeds.

### Matriz e execução

- Formato único desta rodada: `10k×100`; tamanhos menores e maiores foram
  removidos da comparação a pedido do mantenedor.
- Cenários: ruído puro, sinal forte, sinais fracos, correlação, redundância,
  interações e desbalanceamento.
- Fixar datasets, seeds, commits, árvores, threads, hardware e versões.
- Medir iterações fixas e execução até o stopping próprio separadamente.
- Cinco execuções por configuração para desempenho/RSS; três seeds e duas
  iterações por cenário sintético para a checagem exploratória de qualidade.
  O gate de qualidade para conclusões fortes continua sendo trinta seeds.
- Registrar wall time, CPU, RSS, iterações, dimensão ativa, precisão/recall da
  seleção, tentativas e desempenho preditivo posterior.
- Medir conversão separadamente e incluí-la também no tempo ponta a ponta.
- Não incluir compilação no tempo de fit; registrar warm-up e ordem de execução.

### Critérios de aceite

- Decisões corretas nos oráculos e paridade virtual/materializada.
- Nenhuma cópia integral de `X` no caminho estrito; nenhuma reconstrução da
  matriz aumentada por iteração.
- Sem crescimento de memória proporcional a `Tnp`; histórico desligado.
- Suítes dos dois projetos aprovadas e artefatos instaláveis.
- Meta de razão geométrica de tempo abaixo de `1,0` contra sklearn e XRF na
  comparação controlada; publicar também resultados individuais desfavoráveis.
- Modo agressivo deixa de ser experimental somente se o limite inferior dos
  intervalos pareados de 95% da diferença agressivo menos canônico for maior
  que `-0,02`, tanto para precisão quanto recall nos cenários com relevantes.
  No cenário nulo, avaliar falsas seleções separadamente, sem recall indefinido.
- Não declarar liderança de velocidade sem evidência contra concorrentes nativos.
- Cada marco concluído deve estar documentado, commitado e enviado ao GitHub.

## Fundamentação e referências

Inspeção inicial do motor em `bankai-random-forest-v2`, revisão
`b29a7fc07fc36405a17c7739aaefef99baa72d79`: o binding `matrix_to_owned` copia a
matriz NumPy; o agregador de importância copia rótulos por árvore. Esses caminhos
motivam a extração do núcleo e os testes de memória. O motor já contém suporte
a descarte de árvores e agregação de importância, que deve ser reaproveitado.

- [Boruta-Shap — código](https://github.com/Ekeany/Boruta-Shap/blob/master/src/BorutaShap.py):
  referência de flexibilidade; materializa shadows e concatena DataFrames.
- [boruta-fru — implementação Rust](https://docs.rs/boruta-fru/0.1.0/src/boruta_fru/lib.rs.html):
  loop nativo e preservação de confirmadas; concorrente direto de benchmark.
- [Boruta R — política estatística](https://github.com/cran/Boruta/blob/master/R/Boruta.R):
  binomial, Bonferroni e shadows; fixar revisão nas fixtures de implementação.
- [Arrow PyCapsule](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html):
  protocolo para importação dos buffers colunares.

O link Credly fornecido não estava acessível durante a análise e não foi usado
como fundamento técnico.
