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

**Status:** pendente.

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

- [ ] Fixtures determinísticas e contratos executáveis aprovados.
- [ ] Harness de benchmarks e dados sintéticos reproduzíveis disponíveis.
- [ ] Verificações concluídas, roadmap atualizado, commit e push realizados.

**Commit:** `test: establish boruta contracts and benchmark harness`

## M2 — Núcleo Rust compartilhado

**Status:** pendente.

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

- [ ] Crate compartilhado sem dependência de Python.
- [ ] Suítes dos dois projetos aprovadas.
- [ ] Alteração do motor commitada e enviada ao seu remoto verificado.
- [ ] Boruta fixa a revisão publicada, sem caminhos absolutos.
- [ ] Roadmap atualizado, commit e push do Boruta realizados.

**Commit:** `refactor: extract shared native forest core`

## M3 — NumPy zero-copy e shadows virtuais

**Status:** pendente.

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

- [ ] Paridade numérica com o oráculo materializado.
- [ ] Zero cópias integrais de `X` comprovadas no caminho estrito.
- [ ] Memória auxiliar medida.
- [ ] Verificações concluídas, roadmap atualizado, commit e push realizados.

**Commit:** `feat: add zero-copy numpy input and virtual shadows`

## M4 — Boruta canônico completo

**Status:** pendente.

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

- [ ] Decisões iguais ao oráculo para as mesmas sequências de importância.
- [ ] Limite de iterações exato e ausência de resolução heurística de tentativas.
- [ ] Seletor funcional ponta a ponta.
- [ ] Verificações concluídas, roadmap atualizado, commit e push realizados.

**Commit:** `feat: implement canonical native boruta selection`

## M5 — Poda agressiva e paralelismo determinístico

**Status:** pendente.

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

- [ ] Invariantes de poda e reprodutibilidade aprovadas.
- [ ] Tempo, precisão, recall e estabilidade publicados de forma pareada.
- [ ] Modo permanece experimental até satisfazer o critério de qualidade.
- [ ] Verificações concluídas, roadmap atualizado, commit e push realizados.

**Commit:** `perf: add aggressive pruning and deterministic parallelism`

## M6 — Polars, sklearn e ciclo de vida

**Status:** pendente.

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

- [ ] NumPy obrigatório e Polars opcional, com contratos documentados.
- [ ] Exemplos incluem seleção dentro de Pipeline.
- [ ] Serialização compacta e sem reconstrução de floresta.
- [ ] Verificações concluídas, roadmap atualizado, commit e push realizados.

**Commit:** `feat: add polars and sklearn selector integration`

## M7 — Benchmarks, endurecimento e distribuição

**Status:** pendente.

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

- [ ] Protocolo abaixo executado com resultados brutos e relatório versionados.
- [ ] Suítes dos dois projetos aprovadas e distribuições instaláveis.
- [ ] Limitações e perdas de desempenho registradas, sem seleção de resultados.
- [ ] Artefatos construídos exclusivamente com Maturin.
- [ ] Roadmap atualizado, commit e push realizados.

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

Na comparação controlada com sklearn, usar importância por árvore em dados
OOB no harness. Não tratar `sklearn.inspection.permutation_importance` sobre
outro conjunto como a mesma operação. Identificar explicitamente Gini/SHAP e
demais diferenças nas comparações de soluções completas.

### Matriz e execução

- Formatos: `1k×20`, `10k×100`, `100k×100` e `2k×5k`.
- Cenários: ruído puro, sinal forte, sinais fracos, correlação, redundância,
  interações e desbalanceamento.
- Fixar datasets, seeds, commits, árvores, threads, hardware e versões.
- Medir iterações fixas e execução até o stopping próprio separadamente.
- Cinco execuções por configuração para desempenho/RSS; trinta seeds por
  cenário sintético para qualidade.
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
