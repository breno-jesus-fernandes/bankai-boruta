# M8 — Escopo e limite da verificação Miri

Em 2026-09-27, Miri passou pelos testes de `bankai-arrow-ffi` (3/3) e pelos
targets `boruta_loop`, `decision_ledger`, `matrix_view` e
`native_importance_provider` de `boruta-core`. O adapter sklearn não introduziu
`unsafe` e não alterou os caminhos Arrow FFI.

O comando amplo `cargo +nightly miri test -p boruta-core` foi interrompido
durante `shared_forest_engine::borrowed_and_owned_forest_importance_match_across_seeds_and_class_counts`,
após mais de 20 minutos interpretando treinos repetidos de floresta no XRF
upstream. Miri não apontou UB antes da interrupção; a execução foi encerrada
por custo de tempo, não por falha do teste. A suíte Cargo normal passou nesse
target, incluindo todos os seeds e contagens de classe.

O gate de segurança do M8 cobre os caminhos `unsafe` locais: Arrow FFI passou
integralmente em Miri e o adapter Rust/Python não usa `unsafe`. Se o código de
`bankai-forest-core`/XRF upstream mudar, rode o target interrompido isolado e
avalie uma fixture reduzida para manter a análise Miri viável.
