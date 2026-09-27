# M6: limitação Arrow zero-copy resolvida

**Status:** resolvida em 2026-09-27 no M6. NumPy segue emprestado como antes;
Polars agora é aceito diretamente via Arrow C Stream, sem conversão NumPy,
rechunk ou cópia dos buffers numéricos importados. As garantias detalhadas,
limites de tipos e contrato de ownership estão em
[`crates/arrow-ffi/SAFETY.md`](../../crates/arrow-ffi/SAFETY.md).

O primeiro protótipo com pyo3-arrow foi descartado depois que o teste de buffer
desalinhado mostrou que arrow-rs pode corrigir alinhamento copiando. O adaptador
atual é pequeno e isolado: valida esquema, comprimentos, offsets e alinhamento
antes de transferir os owners para os tipos Arrow RAII. Colunas Polars são
importadas individualmente porque o exportador do DataFrame consolida chunks
com fronteiras distintas; os streams das Series preservam chunks e offsets.

O modo estrito aceita Float32/Float64. Nulls, valores não finitos, tipos
incompatíveis, schemas com extensões, colunas duplicadas, overflow e buffers
desalinhados falham com `ValueError`. Inteiros seguem disponíveis por
`input_mode="copy"`. O extra Polars é opcional. Exportadores terceiros podem
fazer suas próprias cópias antes de chamar o importador; veja a definição
medida de `copied_bytes` no README.

Testes Rust confirmam paridade de armazenamento, uma liberação do stream e
retorno dos owners de buffers ao baseline no sucesso e no erro parcial. Miri
passou nos três testes do crate FFI isolado. O teste Miri do binding PyO3 maior
não foi executado: a compilação esgotou o espaço temporário da máquina antes
de iniciar o runner. As regressões funcionais e o build do binding passaram
normalmente. Os resultados completos estão em
[`benchmarks/results/m6_validation_2026-09-27.json`](../../benchmarks/results/m6_validation_2026-09-27.json).

Plano e execução: [`M6-resolution-plan.md`](M6-resolution-plan.md).
