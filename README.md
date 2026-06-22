# benchmark_local

Resultados de benchmark local de modelos GGUF num painel simples.

## Fluxo

1. Coloque `.gguf` em `models/`.
2. Rode o PowerShell do projeto:

```powershell
.\run-benchmark-auto.ps1
```

3. Ao final, escolha uma das opções para visualizar o resultado:

```text
A) Abra o arquivo index.html (na raiz do repo) e clique em Carregar JSON para subir reports/benchmark_results.dashboard.json.
B) Para o caminho HTTP: python -m http.server 8000
```

## Saídas geradas

- `reports/benchmark_results.csv` — registros brutos
- `reports/benchmark_results.dashboard.json` — payload do dashboard

Arquivos em `models/` e `reports/` não são enviados para o GitHub.

## Dashboard

O painel usa `index.html` com dados embarcados em uma tag inline JSON. Não depende de servidor HTTP para carregar os resultados. Basta escolher um dos dois modos acima.

## Licença

MIT
