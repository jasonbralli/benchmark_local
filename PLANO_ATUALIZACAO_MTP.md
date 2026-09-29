# Plano de atualização — benchmark_local → llama.cpp b11222 (MTP nativo)

**Data:** 27/09/2026 · **Autor:** Hermes

## 1. Situação atual (medida)

| Item | Valor |
|---|---|
| Build instalado | **b9637** (`aedb2a5e9`, Clang 20.1.8) |
| Última release (27/09/2026) | **b11222** |
| Gap | ~1.585 builds atrás |
| Caminho binário | via PATH (`llama-server.exe`) |

## 2. O que muda (b9637 → b11222, relevo MTP)

- Build **~b9200+**: MTP (Multi-Token Prediction) embutido no próprio GGUF — modelos Qwen3.8-27B, Qwen3.6, Gemma 4 etc. carregam esp. decoding sem `--model-draft` separado.
- Flags **novas/renomeadas** disponíveis já no build atual (validado em `llama-server --help`):
  - `--spec-type {none,draft-simple,draft-eagle3,draft-mtp,ngram-simple,ngram-map-k,ngram-map-k4v,ngram-mod,ngram-cache}`
  - `--spec-draft-n-max N` (default 3)
  - Flags `--draft`, `--draft-n`, `--draft-max` **REMOVIDAS** → usar `--spec-draft-n-max` / `--spec-ngram-mod-n-max`.
- b10217+: `load_mtp` em `llama_model_params` — MTP head só carrega quando especulativo ativo (zero custo para chamadas não-especulativas).
- **Atenção:** `--draft`, `--draft-n`, `--draft-max` foram REMOVIDOS — qualquer script legado que os use quebra no b11222.

## 3. Impacto no benchmark_local

### Hoje (estado do código)
- `run-benchmark-auto.ps1` e `scripts/benchmark_models.py` **NÃO** usam `--model-draft`.
- Nenhuma referência a `spec-*`/`draft`/`mtp` nos scripts (verificado com grep).
- Server sobe com `--ctx-size 65536` + ngl padrão; MTP nunca foi exercitado.

### Após atualização, para "ler modelo com MTP embutido"
Lançar com:
```
llama-server -m <modelo-com-mtp.gguf> --spec-type draft-mtp --spec-draft-n-max 2 ...
```
O projeto precisa:

1. **Passar flags MTP ao `llama-server`** (`benchmark_models.py` + `run-benchmark-auto.ps1`)
2. **Medir acceptance rate** — capturar `draft model type = mtp` e `predicted_tokens_seconds` do log/metrics.
3. **(Opcional)** expor novas métricas no dashboard: `spec_acceptance`, `tokens_speculated`, `tps_com_mtp` vs baseline.

## 4. Plano de execução (incremental)

### Fase 0 — Atualizar binário
```powershell
winget upgrade llama.cpp
# ou manual: baixar b11222 de github.com/ggml-org/llama.cpp/releases
```
Validar: `llama-server --version` → espera-se ≥ b10200 (MTP estável).

### Fase 1 — Suporte MTP no launcher (`benchmark_models.py`)
- [ ] Novo arg CLI: `--spec-type draft-mtp` (default `none` para não quebrar flows)
- [ ] Novo arg: `--spec-draft-n-max 2` (default conservador p/ RTX 5060 Ti; validado comunidade ago/2026)
- [ ] Novo arg: `--mmproj <path>` + `--image-min-tokens 1024` (modelos Qwen-VL/MTP multimodal)
- [ ] Propagar `--spec-*` extras via `--server-arg` existente (escapa para casos avançados)

### Fase 2 — Captura de métricas MCP
- [ ] `load_log_metrics()` em `benchmark_models.py`: parsear
  - `speculative decoding: draft model type = mtp`
  - linha `MTP draft` no banner (alocação VRAM do head)
  - `predicted_tokens_seconds` do endpoint `/metrics`
- [ ] Adicionar campos ao `BenchmarkSample`: `spec_type`, `spec_n_max`, `spec_accept_rate`, `tps_spec`, `tps_baseline_delta`
- [ ] Persistir em `dashboard.json` + `results.csv`

### Fase 3 — Atualizar `run-benchmark-auto.ps1`
- [ ] Aceitar `-SpecType` e `-SpecNMax`
- [ ] Por padrão: detectar se modelo tem `mtp_num_hidden_layers>0` no GGUF metadata — se sim, aplicar `draft-mtp` automaticamente (aviso no log)
- [ ] Validar PS1 salvo em UTF-8 com BOM (regra do projeto)

### Fase 4 — Dashboard (`index.html`)
- [ ] Nova coluna/tabela: `MTP` (badge on/off), `n_max`, `accept %`, `Δ t/s`
- [ ] Opcional: gráfico comparando modelo com/sem MTP no mesmo setup
- [ ] Assinatura InovaTudo no footer já presente (manter)

### Fase 5 — Testes
- [ ] `tests/test_benchmark.py`: caso `spec_type=draft-mtp` → assert que arg propaga p/ `llama-server`
- [ ] Smoke real com `Qwen3.8-27B-UD-IQ3_S.gguf` (já validado na máquina, 47 t/s ctx 65536):
  - `--spec-type draft-mtp --spec-draft-n-max 2` → esperar log `draft model type = mtp, n_max = 2`
  - Medir `predicted_tokens_seconds` antes/depois

## 5. Rollback
- Se b11222 quebrar algo → reverter para b9637 (mantido em `%LOCALAPPDATA%/llama.cpp/backup/` ou via winget pin).
- Flags MTP são opt-in (`--spec-type none` é o padrão) → comportamento legacy preservado.

## 6. Critério de aceite
1. Build ≥ b10200 instalado
2. `benchmark_models.py --spec-type draft-mtp --spec-draft-n-max 2 -m Qwen3.8-27B-UD-IQ3_S.gguf` sobe o server com log `speculative decoding: draft model type = mtp, n_max = 2`
3. `pytest tests/` passa
4. Dashboard exibe `spec_accept_rate` para a amostra MTP
5. Aceitação medida ≥ 0.5 em geração de código (referência comunidade)

## 7. Riscos / cuidados
- RTX 5060 Ti 16GB: MTP draft adiciona ~1.3 GB VRAM no Qwen3.8-27B (KV q4_0) — pode estourar com ctx grande; usar `--cache-type-k q4_0 --cache-type-v q4_0`.
- `--flash-attn off` + `--cache-type-v q4_0` → inválido (`V cache quantization requires flash_attn`).
- Build b11222 é bleeding-edge: se aparecer regressão, rebaixar para **b10240** (1ª que carrega `load_mtp` do PR #26296).
