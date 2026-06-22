# Benchmark Local de Modelos GGUF

Este projeto testa modelos GGUF locais, um por vez, mede desempenho e registra sinais de memória e contexto para comparação prática em hardware local.

## Estrutura

```
.
├─ models/          # modelos locais, ignorados pelo Git
├─ reports/         # resultados locais, ignorados pelo Git
│  └─ server-logs/  # logs dos servidores por modelo
├─ scripts/
│  ├─ benchmark_models.py
│  ├─ benchmark_scorer.py
│  └─ prompts.json
├─ index.html
├─ run-benchmark-auto.ps1
├─ .gitignore
└─ LICENSE
```

## Como usar

1. Coloque os modelos `.gguf` em `models/`.
2. Rode o benchmark:

```powershell
.\run-benchmark-auto.ps1
```

Ao final, o dashboard abre automaticamente em **http://localhost:8000/** com os dados carregados.

## O que o benchmark faz

- encontra os `.gguf` dentro de `models/`, inclusive em subpastas
- sobe um `llama-server` por modelo
- espera o servidor ficar pronto
- faz warmup
- executa a bateria de prompts
- mede:
  - `elapsed_seconds`
  - `tokens_generated`
  - `tokens_per_second`
  - tamanho do arquivo do modelo
  - contexto usado
  - `server_ctx_train`
  - estado observado no log do servidor
- gera relatórios em `reports/`

## Saídas geradas

- `reports/benchmark_results.csv`
- `reports/benchmark_results.dashboard.json`
- `reports/dashboard_data.js`
- `reports/server-logs/*.log`

Os arquivos em `models/` e `reports/` não são enviados ao GitHub.

## Dashboard

O arquivo `index.html` é servido pelo próprio `run-benchmark-auto.ps1` em `http://localhost:8000/`. O painel carrega os dados embarcados em `reports/dashboard_data.js` para evitar bloqueios de `file://`.

O painel tem três blocos:
- comparativo simples por modelo e categoria
- detalhes técnicos com métricas de memória e contexto
- pontuação e estatísticas avançadas

## Licença

MIT
