# Plano de conclusão do M6 — Polars/Arrow sem cópia

Status: implementado e validado em 2026-09-27. Este plano registra decisões,
testes e os critérios usados para encerrar o M6. A nota de limitação contém o
resumo e os limites remanescentes.

## Diagnóstico do código atual

- `_estimator.py::fit` rejeita entradas não NumPy no modo estrito e chama
  `np.asarray(X)` antes do binding. Esse despacho deve mudar antes da conversão.
- `MatrixView<T>` depende de um único `&[T]` e strides. Não representa buffers
  independentes, tipos mistos ou chunks por coluna.
- `MatrixForestInput`, `ImportanceProvider<T>` e `run_boruta` dependem dessa
  representação. A generalização deve preservar splitter, RNG, shadows e
  inferência estatística já testados.
- No modo copy, uma entrada Polars pode ser materializada em `np.asarray` e
  novamente no `numpy.array(copy=True)` do binding. Remover a duplicação como
  parte da integração, preservando a contabilização das cópias reais.
- O core e o binding proíbem `unsafe`. Preferir importador com API segura;
  eventual FFI próprio deve ficar isolado, revisado e testado, sem relaxar a
  proibição global dos crates existentes.
- Os testes atuais de pickle só excluem alguns nomes de atributos. Eles não
  demonstram ausência de retenção por outros objetos nem tamanho independente
  do número de amostras. O smoke test instalado não cobre Polars ou joblib.

## Contrato de conclusão

`BankaiBoruta.fit(X, y)` deve aceitar NumPy como hoje e Polars numérico via Arrow
C Stream, com uma única chamada ao motor e todas as iterações em Rust.

No modo estrito, suportar inicialmente Float32/Float64, inclusive misturados,
colunas com múltiplos chunks e slices com offsets. Rejeitar nulls, NaN/Inf,
strings, categorias, nested/extension types e inteiros com mensagem específica;
`input_mode="copy"` continua sendo a rota explícita para conversão numérica.
Inteiros nativos podem ser acrescentados depois com política de precisão para
valores fora da faixa de representação exata de f64. Não converter silenciosamente.

Zero-copy significa nenhum buffer de valores de X duplicado no ingresso ou
por iteração. Metadados, índices shadow, labels e workspaces são alocações
separadas e permitidas. Um exportador arbitrário pode copiar antes de entregar
Arrow: declarar a garantia ponta a ponta somente para versões/layouts Polars
verificados, e a garantia do importador separadamente.

Manter o GIL no primeiro incremento e os owners vivos até todos os workers
terminarem. Não executar callbacks Python dentro do loop de treino. Liberar o
GIL é trabalho posterior, condicionado à auditoria de ownership e callbacks.

## M6.1 — Prova de compatibilidade e contrato de entrada

**Objetivo:** escolher uma ponte Arrow compatível antes de refatorar o core.
Investigar `pyo3-arrow`/arrow-rs com o PyO3 0.29.2 e abi3-py311 existentes;
compatibilidade ainda não verificada. Fixar versões após o spike. Importar
`__arrow_c_stream__` sem passagem por `to_arrow()`/NumPy ou dependência Python
obrigatória de PyArrow. PyArrow pode ser dependência exclusiva dos testes.

**TDD Rust:** schema primitivo válido, tipo inválido, stream vazio, erro ao ler
um batch e validação de offsets/comprimentos sem desreferenciar memória inválida.
**TDD Python:** exporter de teste cuja conversão NumPy falha; cápsula com nome
inválido; versões mínima e corrente de Polars; ausência de Polars não impede
importar/usar NumPy. Testes de callback/release usam produtores válidos e
instrumentados, não ponteiros arbitrários.

**Memória:** obter buffers e owners sem copiar valores; verificar endereços e
offset efetivo em fixtures. Não solicitar cast de schema ou rechunk automático.
**Riscos:** conflito PyO3/abi3 e importador que realinha/copia silenciosamente.
**Saída:** protótipo compilável por Maturin, versões fixadas e decisão registrada.
Se nenhum importador seguro for compatível, avaliar crate FFI mínimo isolado
com auditoria; não espalhar raw pointers no algoritmo.

## M6.2 — Separar armazenamento e view lógica

**Objetivo:** introduzir uma abstração de leitura (`FeatureStorage`: shape e
leitura por coordenada), com adaptador strided NumPy e adaptador colunar.
Generalizar a view lógica, o input da floresta, o provider e o loop sobre essa
abstração. Preferir dispatch estático no caminho NumPy; manter shadows e IDs
originais acima do armazenamento. Não duplicar splitter ou algoritmo Boruta.

**TDD Rust:** paridade strided/colunar célula a célula; shadows virtuais contra
matriz materializada; decisões, hits e importâncias seeded; tipos f32/f64;
paridade serial/paralela; poda com IDs originais. Reutilizar os oráculos atuais.
**TDD Python:** todos os testes existentes de C/F, zero-copy, uma entrada FFI,
seleção e determinismo continuam passando sem alteração de expectativas.

**Memória:** clones de handles/metadados não copiam buffers. A abstração de
leitura não deve exigir `Vec<f64>` nem materializar a matriz aumentada.
**Riscos:** dispatch por célula, bounds checks extras e mudança de RNG/ordem.
**Saída:** backend NumPy preservado e backend colunar testável sem Python.

## M6.3 — Colunas fragmentadas e ownership Arrow

**Objetivo:** adicionar armazenamento de chunks com owners RAII e metadados
por coluna; tratar offsets usando a API Arrow, sem aplicá-los duas vezes.

**TDD Rust:** chunks vazios/intercalados, fronteiras, slices não alinhados entre
colunas, f32/f64 misturados, dimensões inconsistentes, overflow e nulls; contagem
de release de cada recurso exatamente uma vez no sucesso e nos erros parciais;
clones/views mantêm owners válidos até o último uso. Testar leitura por workers.
**TDD Python:** slices e concatenação sem rechunk; erro tardio no stream libera
batches anteriores; produtor sai de escopo após importação em fixture de teste
sem invalidar o owner importado.

**Memória:** armazenar referências a batches/chunks e offsets cumulativos,
O(p + K) metadados para K chunks de coluna. Boruta exige acesso aleatório: reter
os batches importados, sem alegar processamento out-of-core. Inicialmente busca
binária do chunk O(log K_coluna), com caminho direto para coluna de um chunk.
Não adicionar tabela O(n*p) para resolver linhas.
**Riscos:** callback no thread errado, double-free e custo de lookup em dados
muito fragmentados. Drenar o stream e destruir owners Python no contexto seguro;
workers apenas leem buffers, e precisam terminar antes da liberação final.
**Saída:** paridade com matriz materializada e testes de lifetime aprovados.

## M6.4 — Integrar fit, diagnóstico e contrato sklearn

**Objetivo:** despachar NumPy/Arrow antes de qualquer conversão no Python;
passar o objeto à única chamada nativa. Validar y e metadados sem materializar X.
Integrar nomes de features e forma diretamente do schema. Simplificar copy mode
para uma materialização de valores quando tecnicamente possível.

**TDD Python:** monkeypatch em `DataFrame.to_numpy`/`__array__` que falha no modo
estrito; NumPy e Polars geram mesmos hits/decisões para mesmo conteúdo e seed;
nomes/ordem, duplicatas inválidas, n_jobs, nulls e schema inválido. Testar
`Pipeline` com classificador e validação cruzada, seleção dentro de cada fold,
`get_support`, `transform`, `get_feature_names_out`, `set_output` quando suportado,
refit entre formatos e política explícita de estado após fit que falha.
**TDD Rust:** nenhum caminho Arrow chega ao adaptador strided por conversão;
validação e diagnósticos coerentes em sucesso/erro.

**Memória:** separar bytes copiados de X, metadados e índices auxiliares.
`copied_bytes=0` sozinho não comprova zero-copy: combinar identidade dos buffers,
proibição de conversões e instrumentação das alocações. `transform` pode alocar
sua saída; essa permissão não se aplica a `fit`.
**Riscos:** validação sklearn materializar X e dois buffers no copy mode.
**Saída:** `fit(Polars, input_mode="zero_copy")` funcional com contrato publicado.

## M6.5 — Retenção, serialização e artefatos instalados

**Objetivo:** fechar ciclo de vida e dependências opcionais nos artefatos reais.

**TDD Rust:** contadores de owners/alocações voltam ao baseline após fits e erros
repetidos; executar Miri no código seguro compatível e sanitizer nos testes FFI
quando disponível. Registrar explicitamente os ambientes não cobertos.
**TDD Python:** pickle/joblib com histórico desligado têm tamanho praticamente
independente de n; weakrefs/release counters mostram liberação de X; restaurar
estimador após coletar dados originais; transform e nomes preservados. Exercitar
mesmos casos instalando wheel e sdist em ambiente isolado, com e sem extra Polars.

**Memória:** medir RSS em subprocessos repetidos, após warm-up e com orçamento
de ruído definido antes de executar. Não confundir cache do allocator com leak;
combinar RSS com contagem objetiva de owners. Não serializar streams ou forests.
**Riscos:** testes omitidos por `importorskip`, dependências implícitas e retenção
em closures/diagnósticos. Job obrigatório com Polars instalado deve executar os
casos Arrow sem skips.
**Saída:** suite e serialização verificadas nos pacotes Maturin.

## M6.6 — Gate de desempenho e encerramento

**Objetivo:** medir o custo da generalização e do ingresso, isolado do M7 completo.
Comparar NumPy C/F, Polars um chunk, slices e 8/64 chunks, f32/f64/misto,
serial/quatro workers. Fixar seeds, conteúdo, iterações e configurações.

**Testes/experimentos primeiro:** fixtures pequenas verificam igualdade antes de
cronometração; cinco execuções por configuração com ordem alternada e warm-up.
Usar ao menos 1k×20 e 10k×100; reportar conversão/exportação, fit, RSS e cópias.
Comparar NumPy com o commit pré-refatoração. Gate proposto: investigar regressão
reproduzível acima de 10% na média geométrica de tempo NumPy; qualquer aceitação
de regressão exige justificativa registrada, nunca ocultar casos desfavoráveis.
Não prometer que Arrow fragmentado será mais rápido que NumPy contíguo.

**Aceite final obrigatório:**

- [x] M6.1–M6.5 verdes; comportamento estatístico preservado.
- [x] Buffers Polars suportados compartilhados, sem concatenação/rechunk/cast ocultos.
- [x] Owners liberados corretamente em sucesso, erro e execução paralela.
- [x] NumPy obrigatório/Polars opcional; Polars 1.32.0 mínimo e 1.44.2 local testados.
- [x] fmt, clippy estrito, suites Rust/Python e build Maturin aprovados.
- [x] Resultados brutos e relatório versionados; nenhuma alegação de ganho sem evidência.
- [x] Limitação atualizada com evidências de resolução e M6 concluído no roadmap.
- [x] Commits incrementais registrados; checkpoint final será publicado após esta validação.

Cada incremento deve ser um commit revisável após seus testes. O gate técnico do
M6 fechou em 2026-09-27; resta publicar o checkpoint final. A tentativa Miri
do crate PyO3 completo falhou antes do runner por falta de espaço temporário;
Miri passou no crate isolado que contém o unsafe. O benchmark completo contra sklearn/XRF e a matriz ampliada de
plataformas continuam no M7; não são pré-requisitos para desenvolver o adaptador.

## Referências verificadas para o desenho

- [Polars: __arrow_c_stream__](https://docs.pola.rs/api/python/dev/reference/dataframe/api/polars.DataFrame.__arrow_c_stream__.html)
  documenta exportação por cápsula; verificar versões efetivas no spike.
- [Arrow PyCapsule Interface](https://arrow.apache.org/docs/format/CDataInterface/PyCapsuleInterface.html)
  define consumo único e responsabilidades de release.
- [pyo3-arrow](https://docs.rs/pyo3-arrow/latest/pyo3_arrow/) foi avaliado no
  spike, mas descartado após detectar possível cópia ao reparar alinhamento.
