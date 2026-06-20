# Benchmark Local de Modelos GGUF

Este projeto testa modelos GGUF locais, um por vez, mede desempenho e registra sinais de memória e contexto para comparação prática em hardware local.

## Estrutura

```text
.
├─ README/
│  ├─ README.md
│  └─ prompts.example.json
├─ index.html
├─ models/          # modelos locais, ignorados pelo Git
├─ reports/         # resultados locais, ignorados pelo Git
├─ scripts/
│  ├─ benchmark_models.py
│  └─ run-server.ps1
├─ run-benchmark-auto.ps1
├─ .gitignore
└─ .git
```

## Como usar

1. Coloque os modelos `.gguf` em `models/`.
2. Rode o benchmark:

```powershell
.\run-benchmark-auto.ps1
```

3. Abra o painel:

```text
./index.html
```

Se quiser servir a pasta via HTTP:

```powershell
python -m http.server 8000
```

Depois acesse:

```text
http://localhost:8000/
```

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

## Arquivos gerados

- `reports/benchmark_results.csv`
- `reports/benchmark_results.summary.csv`
- `reports/benchmark_results.ranking.csv`
- `reports/benchmark_results.dashboard.json`
- `reports/server-logs/*.log`

Os arquivos em `models/` e `reports/` não devem ser enviados ao GitHub. O repositório mantém apenas a estrutura com arquivos `.gitkeep`.

## Métricas de memória

Os campos mais importantes são:

- `model_size_bytes`: tamanho do arquivo em disco
- `context_size`: janela de contexto usada no teste
- `server_ctx_train`: contexto de treino reportado pelo servidor
- `observed_state_size_bytes`: maior estado observado no log do servidor
- `estimated_kv_cache_bytes`: estimativa de KV cache usada pelo benchmark
- `estimated_loaded_bytes`: soma do arquivo com a carga de contexto observada ou estimada

Essas métricas ajudam a comparar não só velocidade, mas também viabilidade local.

## Dashboard

O arquivo `index.html` lê o JSON de `reports/benchmark_results.dashboard.json`.

O painel tem dois blocos:

- um comparativo simples por modelo e categoria
- um bloco técnico com as métricas detalhadas

## Sobre o GitHub

A organização atual já está pronta para versionamento:

- `scripts/` para automação
- `README/` para documentação e exemplos
- `models/` para seus modelos locais
- `reports/` para saídas geradas, normalmente ignoradas pelo Git

Antes de subir, vale conferir:

- se os modelos em `models/` devem ser versionados ou não
- se você quer manter os `reports/` fora do repositório
- se o caminho do `llama-server.exe` precisa ser parametrizado por máquina
