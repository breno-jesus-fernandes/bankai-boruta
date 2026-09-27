# M6: limitação de importação Polars/Arrow sem cópia

O adaptador atual foi projetado para emprestar diretamente matrizes NumPy
contíguas e alinhadas. Polars pode ser usado em `input_mode="copy"`: a conversão
para NumPy é explícita no contrato e os bytes materializados são adicionados a
`diagnostics_["copied_bytes"]`. `input_mode="zero_copy"` rejeita Polars.

O teste de integração cobre a conversão Polars -> NumPy, nomes de colunas e
contabilização do buffer. Ele não prova importação zero-copy dos buffers Arrow.
Para suportar essa rota corretamente é necessário desenhar o ownership/lifetime
do protocolo Arrow C Data/C Stream através do PyO3, incluindo chunks, offsets,
nulls, alinhamento, tipos numéricos e liberação única dos owners. `to_numpy()`
não satisfaz esse requisito e pode materializar colunas mesmo quando algum caso
isolado parece contíguo.

Até essa implementação e os testes de lifetime, manter Polars como extra
opcional com cópia declarada. Não alegar zero-copy para esse caminho. Os testes
de ciclo de vida sklearn e os casos explícitos de cópia estão passando; a
conclusão de M6 permanece pendente desta limitação e da verificação da
serialização em artefatos instaláveis.
