# benchmark_local

Resultados de benchmark local de modelos GGUF em um painel unificado e autossuficiente.

## Fluxo

1. Coloque `.gguf` em `models/`.
2. Rode o PowerShell do projeto:

```powershell
.\run-benchmark-auto.ps1
```

3. Ao final, abra `index.html` e clique em **Carregar JSON** para subir `reports/benchmark_results.dashboard.json`.

> O dashboard embute o JSON diretamente no HTML, sem depender de servidor HTTP.

## Saídas geradas

- `reports/benchmark_results.csv` — registros brutos
- `reports/benchmark_results.dashboard.json` — payload do dashboard
- `reports/scoring_summary.json` — resumo de avaliação por modelo

Arquivos em `models/` e `reports/` não são enviados para o GitHub.

## Dashboard

O painel compara até 4 blocos:

- **Performance** — velocidade média por modelo e por categoria
- **Qualidade** — score final, taxa de sucesso e avaliação textual
- **Consistência** — variância, min/max tok/s, tempo e tokens
- **Recursos** — tamanho do modelo, KV cache, carga estimada e contexto

Todos os blocos compartilham a mesma ordenação global. Basta clicar no cabeçalho de qualquer coluna para reorganizar todos os blocos juntos.

## Licença

MIT
