# benchmark_local

Resultados de benchmark local de modelos GGUF em um painel unificado e autossuficiente.

## Fluxo

1. Coloque `.gguf` em `models/`.
2. Rode o PowerShell do projeto:

```powershell
.\run-benchmark-auto.ps1
```

3. Ao final, abra `index.html` e clique em **Carregar JSON** para subir `reports/benchmark_results.dashboard.json`.

> O dashboard embute o JSON no HTML, mantendo tudo em arquivos estáticos, sem depender de servidor HTTP.

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

## Recursos visuais e usabilidade

- **Tooltips nos cabeçalhos dos blocos**: ao passar o mouse sobre *Performance*, *Qualidade*, *Consistência* ou *Recursos*, aparece uma descrição curta daquele bloco.
- **Tooltips nas colunas das tabelas**: ao passar o mouse sobre cada título de coluna, uma dica explica rapidamente o que aquela métrica significa.
- **Ordenação por coluna**: clique no título de qualquer coluna para ordenar a tabela.
- **Indicador visual da ordenação**: a coluna ordenada exibe uma seta (`▲` ou `▼`), mostrando a direção atual sem precisar advinhar.
- **Ordenação correta por categoria**: as colunas *coding*, *extraction*, *instruction* e *reasoning* agora usam o valor real dentro de `categories.<nome>.avg_tokens_per_second`, garantindo ordenação correta em vez de considerar a posição da linha.
- **Exportar PDF**: o botão **Exportar PDF** usa a impressão nativa do navegador com estilo dedicado, permitindo salvar ou imprimir a visão atual do dashboard em um documento.

## Licença

MIT
