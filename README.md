# benchmark_local

Benchmarks de modelos locais GGUF em um painel unificado e autossuficiente.

> **English** — benchmark_local is an automated benchmarking pipeline for open-weights LLMs (GGUF) served locally with [llama.cpp](https://github.com/ggml-org/llama.cpp): one server per model, a 20-prompt suite across coding / structured extraction / instruction-following / reasoning, automated rubric scoring (speed, quality, consistency, resources) and a static dashboard. It auto-detects Multi-Token Prediction (MTP) heads in the GGUF metadata to enable speculative decoding, and pins 9 validated serving flags so the numbers reflect a tuned, real-world configuration. Full documentation in Portuguese below.

## Descrição

Este projeto avalia automaticamente modelos locais de linguagem (formato GGUF) em quatro categorias de tarefas:
- **Coding** — geração e edição de código Python
- **Extraction** — extração de dados estruturados
- **Instruction** — conformidade com instruções e restrições
- **Reasoning** — raciocínio lógico e matemático

Fornece métricas de performance (tokens/s), qualidade (score automatizado), consistência (variância) e uso de recursos (memória).

## Requisitos

- Python 3.9+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) (`llama-server.exe`) — o pipeline resolve sozinho o build CUDA local em `C:\Users\<user>\llama.cpp\build-cuda\bin\Release\llama-server.exe`, se existir; senão usa a variável de ambiente `LLAMA_SERVER_EXE` ou o PATH
- PowerShell (Windows)

## Instalação e Setup

### 1. Copiar modelos GGUF

Coloque os arquivos `.gguf` na pasta `models/`.

### 2. Executar setup

```powershell
.\setup.ps1
```

O script:
- Verifica se Python 3.9+ está instalado
- Localiza ou pede o caminho do `llama-server.exe`
- Armazena o caminho como variável de ambiente `LLAMA_SERVER_EXE`
- Cria as pastas necessárias (`models/`, `reports/`, `reports/server-logs/`)
- Verifica que os arquivos essenciais estão presentes

### 3. Rodar o benchmark

```powershell
.\run-benchmark-auto.ps1
```

O script:
- Procura `llama-server.exe` (build CUDA local `Release`, variável de ambiente, PATH, ou manual)
- Sobe um servidor por modelo (porta aleatória) com as [9 flags de serving validadas](#flags-de-servidor-validadas)
- Detecta automaticamente cabeças MTP no GGUF e habilita speculative decoding quando o modelo suporta (`--spec-type auto`)
- Realiza warmup para estabilizar o modelo
- Executa 20 prompts distribuídos em 4 categorias (5 por categoria)
- Avalia as respostas com scoring automatizado
- Gera os relatórios

### 4. Reavaliar sem reexecutar (`rescore.py`)

```bash
python3 scripts/rescore.py ..\reports\benchmark_results.csv
```

Recalcula os scores de avaliação usando os prompts e respostas já existentes em `benchmark_results.csv`, sem precisar reexecutar o benchmark completo. Útil para testar mudanças na rubrica de scoring.

### 5. Rodar os testes

```bash
python3 -m pytest scripts/test_benchmark.py -v
```

**65 testes** cobrem todos os scorers, a orquestração do `BenchmarkEvaluator`, limpeza de respostas, consistência de thresholds, o gate de arquiteturas, a propagação das flags `--spec-*`/`--server-arg` e as regressões do probe de servidor (timeout adaptativo).

## MTP e Speculative Decoding (automático)

Modelos recentes embarcam cabeças **MTP** (Multi-Token Prediction) no próprio GGUF — desde o llama.cpp ~b10217 (`load_mtp`) elas carregam junto, sem mais um `--model-draft` separado. O benchmark aproveita isso sem intervenção manual:

- `--spec-type auto` (padrão do `run-benchmark-auto.ps1`): lê o metadata do GGUF (`nextn_predict_layers`) direto do header do arquivo e decide o spec-type efetivo **por modelo**
- Modelos com MTP sobem com speculative decoding (`draft-mtp`, `--spec-draft-n-max 2`); modelos sem MTP rodam com `none` — comportamento legado preservado
- O log do servidor é parseado (`parse_server_log_metrics`) para extrair contexto de carga e estado do spec
- O CSV registra as colunas `spec_type`, `spec_active`, `spec_mtp_layers` e `spec_n_max`, então dá para comparar `com/sem MTP` entre modelos no mesmo relatório

Detalhes técnicos e critérios de aceite: [`PLANO_ATUALIZACAO_MTP.md`](PLANO_ATUALIZACAO_MTP.md).

## Flags de servidor validadas

Sem flags explícitas, o `llama-server` sobe com defaults do build (threads da caixa, múltiplos slots, KV cache f16) — o que causava offload parcial para CPU: **GPU 28% e 14,7 t/s** de geração num RTX 5060 Ti. O `run-benchmark-auto.ps1` fixa 9 flags para um serving reproduzível e todo-GPU:

```
--spec-type auto --spec-draft-n-max 2
--server-arg '-ngl 99'
--server-arg '--flash-attn on'
--server-arg '--cache-type-k q4_0'
--server-arg '--cache-type-v q4_0'
--server-arg '--parallel 1'
--server-arg '--ctx-size 65536'
--server-arg '--batch-size 2048'
--server-arg '--ubatch-size 1024'
--server-arg '--threads <n>'
```

Validado no hardware de referência: pico de **GPU 94%** (14,5 GB VRAM) e **42,7–48 t/s** de geração — cerca de 3× o baseline sem flags.

## Saídas geradas

```
reports/
├── benchmark_results.csv              # Registros brutos de todos os prompts
├── benchmark_results.dashboard.json   # Payload do dashboard (carregável por arrastar)
└── server-logs/                       # Logs do llama-server por modelo
    └── *.server.log
```

Arquivos em `models/` e `reports/` não são enviados para o GitHub.

## Dashboard

O dashboard é um arquivo HTML estático que não depende de servidor HTTP.

### Como usar

1. Abra `index.html` diretamente no navegador
2. Clique em **"Carregar JSON"** e selecione `reports/benchmark_results.dashboard.json`
3. Ou use o botão **"Recarregar"** após o carregamento

### Funcionalidades

- **4 blocos de comparação**: Performance, Qualidade, Consistência, Recursos
- **Ordenação por coluna**: clique no título para ordenar (asc/desc)
- **Tooltips**: explicações ao passar o mouse sobre cabeçalhos e colunas
- **Exportar PDF**: usa a impressão nativa do navegador
- **Indicadores visuais**: setas ▲/▼ mostram a direção da ordenação
- **Badge de timeout**: aviso vermelho na coluna "Timeout" da tabela Performance

### Estrutura do Dashboard

| Bloco | Métricas |
|-------|----------|
| **Performance** | Modelo, Avg tok/s, Coding, Extraction, Instruction, Reasoning |
| **Qualidade** | Modelo, Prompts, Ok, Sucesso (%), Final (score), Avaliação (Excelente/Bom/Aceitável/Fraco) |
| **Consistência** | Modelo, Variância, Min tok/s, Max tok/s, Avg elapsed s, Avg tokens |
| **Recursos** | Modelo, Size, KV cache, Carga GiB, Ctx, Ctx train |

## Arquivos do Projeto

```
benchmark_local/
├── README.md                            # Documentação deste arquivo
├── LICENSE                              # Licença MIT
├── pyproject.toml                       # Metadados e dependências do projeto
├── setup.ps1                            # Script de preparação do ambiente
├── run-benchmark-auto.ps1               # Pipeline principal (serving flags + MTP auto)
├── index.html                           # Dashboard visual (carrega JSON via botão)
├── PLANO_ATUALIZACAO_MTP.md             # Plano da atualização MTP (fechado)
├── PLANO_CORRECAO_PROBE_TIMEOUT.md      # Plano do fix de probe timeout (fechado)
├── models/                              # Pasta para arquivos .gguf (não versionado)
├── reports/                             # Relatórios gerados (não versionado)
│   └── server-logs/                     # Logs do llama-server por modelo
└── scripts/
    ├── benchmark_models.py              # Motor de benchmark (gate de arquiteturas, MTP, métricas)
    ├── benchmark_scorer.py              # Sistema de avaliação automatizada
    ├── rescore.py                       # Re-avaliação offline da rubrica (sem reexecutar)
    ├── regenerate_data.py               # Regeneração do CSV/dashboard.json com nova lógica
    ├── test_benchmark.py                # Suíte pytest (65 testes)
    ├── prompts.json                     # 20 prompts de avaliação (5 por categoria)
    ├── scoring_config.json              # Configuração da rubrica de scoring
    └── run-server.ps1                   # Script auxiliar para subir o servidor
```

## Arquiteturas suportadas

`scripts/benchmark_models.py` mantém um gate `SUPPORTED_ARCHITECTURES` (100+ arquiteturas de LLM: llama, qwen2/3, glm4_moe, hunyuan_moe, minimax_m2, phi3, mamba2, grok etc.). Modelos com arquitetura fora da lista, ou classificados como `NON_LLM_ARCHITECTURES` (encoders, projetores multimodais, embeddings), são pulados com aviso — nunca quebram a execução.

Adicionadas recentemente:
- `k2-horizon` — arquitetura K2 Horizon (IFM); o GGUF grava o nome **com hífen**, o gate aceita as duas grafias
- `modern-bert` — ModernBERT; encoder-only (embeddings/decision models), classificado em `NON_LLM_ARCHITECTURES`, portanto pulado no ranking de LLMs

Adicionar uma arquitetura nova exige patch manual no gate — by design: evita "benchmarcar" algo que o llama-server não suporta de fato.

## Sistema de Avaliação

O benchmark avalia modelos em 4 categorias com pesos diferentes:

| Categoria | Peso | Critérios |
|-----------|------|-----------|
| Coding | 40% | Sintaxe, lógica, eficiência, tratamento de erros, clareza |
| Extraction | 30% | Completude, precisão, formato, estrutura, ausência de alucinações |
| Instruction | 20% | Conformidade com restrições, clareza, relevância, criatividade |
| Reasoning | 10% | Resposta correta, explicação, clareza do raciocínio, justificativa |

**Avaliação final**:
- 🟢 **Excelente** (≥8.5)
- 🟡 **Bom** (≥7.0)
- 🟠 **Aceitável** (≥5.0)
- 🔴 **Fraco** (<5.0)

### Métricas e Fórmulas

#### Tokens por Segundo (TPS)

- **Avg tok/s**: Média aritmética simples de `tokens_generated / elapsed_seconds` em todas as categorias.
- **Avg tok/s Weighted**: Média ponderada por tempo — soma dos tokens gerados multiplicados pelo tempo de execução, dividido pela soma dos tempos. Mais fiel ao uso real: um prompt que demora mais contribui mais para a média.
- **Media**: Média simples entre as 4 categorias (coding, extraction, instruction, reasoning).

#### Timeout

- Quando o modelo atinge o limite de espera (`max_wait = 120s`), o registro é marcado como `timeout: True`.
- No CSV: coluna `timeout` (verdadeiro/falso) e `timeout_count` (quantidade de timeouts por modelo).
- No dashboard: badge vermelho na tabela Performance se o modelo teve qualquer timeout.

#### Sucesso vs Qualidade

- **successful_prompts** no dashboard = prompts com `return_code == 0` (execução completa sem erro).
- **final_score** = avaliação de qualidade (score ≥ 7.0 → "Bom" ou melhor). São métricas diferentes!

### Tratamento de Erros

O benchmark captura e registra automaticamente:
- **Timeout** — modelo excede `max_wait` (120s). `elapsed_seconds = timeout`, sem tokens gerados.
- **Truncated output** — modelo produz menos tokens que o mínimo configurado.
- **Erro no servidor** — `return_code != 0`. O registro é salvo mas o modelo pode não ser avaliado.
- **Erro de leitura** — prompt não encontrado em `prompts.json`. Ignorado com log de aviso.
- **Probe de servidor com timeout adaptativo** — a espera pelo servidor usa `min(60, restante)` do deadline total (`server_start_timeout`), e `_post_json` captura `URLError`, `TimeoutError` e `OSError` com retry — antes, um `TimeoutError` no `getresponse` escapava e derrubava o benchmark no primeiro modelo (kernels Vulkan podem levar >5s no primeiro load). Detalhes em [`PLANO_CORRECAO_PROBE_TIMEOUT.md`](PLANO_CORRECAO_PROBE_TIMEOUT.md).

## Roadmap

- [ ] Suporte a múltiplos modelos em paralelo
- [ ] Comparação visual com gráficos de barras
- [ ] Configuração de prompts customizados via JSON
- [ ] Suporte a outros formatos (GGML, AWQ, EXL2)
- [ ] Integração com GitHub Actions para CI
- [ ] Dashboard com filtros por categoria
- [ ] Exportação de relatórios em PDF com marcação automática
- [ ] Suporte a avaliação com LLM externo (feedback de modelo)
- [ ] Métricas de acceptance rate do MTP (`spec_accept_rate`, `tps_spec` vs baseline)
- [ ] Configuração via `.env` para variáveis de ambiente
- [ ] Suporte a Windows Terminal e WSL

## Licença

MIT License — veja o arquivo [LICENSE](LICENSE).
