# PLANO DE CORREÇÃO — Probe Timeout + MTP falso negativo (benchmark_local)

**Data:** 28/09/2026
**Origem:** auditoria do LOG `Benchmark Local PowerShell 7.6.6-2.txt` (run 28/09 16:46)
**Build:** llama-server 0.5.0-dev **b11193** (winget Vulkan, commit 4e7481175)
**Status:** plano pronto — pendente de implementação

---

## 1. Resumo executivo

A run de 28/09 16:46 falhou com **TimeoutError cru (socket read)** no 1º probe do
`wait_for_server_ready`, logo após o modelo terminar de carregar (26s). O benchmark
inteiro morreu no 1º modelo (GSQ-RCO-IQ3_S) — nenhum resultado salvo.

**2 falhas estruturais identificadas:**

| # | Falha | Severidade |
|---|-------|-----------|
| F-A | `_post_json` não captura `socket.timeout`/`TimeoutError` — só `HTTPError`/`URLError` | **Alta** (mata a run inteira) |
| F-B | `model_has_mtp()` por filename perde MTP embutido real (UD-IQ3_S) | Média (métrica distorcida) |

---

## 2. Evidências

### F1. Timeline do log da run falhada
(`reports/server-logs/Qwen3.8-27B-GSQ-RCO-IQ3_S.server.log`, 12 linhas, mtime 28/09 16:46)

```
0.00.000  initializing
0.25.940  model loaded                    ← carga total 26s
0.25.940  listening on http://localhost:60347
0.26.572  probe chegou (slot LRU)         ← 1º probe do wait_for_server_ready
0.30.039  print_timing prompt processing  ← compilou kernels: 3.47s p/ 3 tokens
                                          (log termina — server derrubado no finally)
```

- 13× `"aguardando o modelo terminar de carregar..."` × 2s = 26s ≈ carga do modelo
  (0.00 → 0.25.940). O servidor **NÃO morreu sozinho** — o `finally` do main o
  derrubou quando a exceção subiu. O print `"servidor finalizado"` é enganoso.
- Traceback: `getresponse → _read_status → readline → recv_into → TimeoutError` —
  timeout de **leitura** da resposta, não de conexão.

### F2. Causa-raiz mecânica (benchmark_models.py:263-283)

```python
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response: ...
    except urllib.error.HTTPError as exc: ...
    except urllib.error.URLError as exc: ...   # ← TimeoutError NÃO é URLError
```

`socket.timeout` (== `TimeoutError` no Python 3.10+) ocorre em `getresponse()`
**durante a leitura** — não é subclasse de `URLError` nesse caminho → **escapa do
try/except** e sobe até `main()` → `SystemExit(1)`.

Por que o 1º probe demorou >5s: compilação de kernels de geração no Vulkan
(recém-carregado 27B IQ3_S) — 3.47s só para prompt processing; a geração de 1 token
somou o restante e estourou o `timeout=5` do probe. Runs anteriores (25/09 UD-IQ3_S
completou 20 prompts) passaram porque o 1º probe casou dentro de 5s — **sorte de
timing, não robustez**.

### F3. MTP falso negativo no UD-IQ3_S (F-B)

| Modelo | Header GGUF | `model_has_mtp()` (filename) | `resolve_spec_type("auto")` |
|--------|-------------|------------------------------|------------------------------|
| Qwen3.8-27B-GSQ-RCO-IQ3_S | sem `nextn_predict_layers` ✓ | False ✓ | `none` ✓ (correto) |
| Qwen3.8-27B-UD-IQ3_S | **TEM `qwen35.nextn_predict_layers`** ✓ | **False** ✗ (stem termina `_S`, não `_mtp`) | `none` ✗ (deveria ser `draft-mtp`) |

Consequência: o benchmark atual mede o UD-IQ3_S **SEM MTP**, inconsistente com o
baseline de 12/09 (47.06 t/s em ctx 65536 **COM** `--spec-type draft-mtp` manual).
O MTP embutido está no arquivo — a heurística por filename não o vê.

### F4. Flags do build b11193 (validadas via `--help`)

| Flag | Status |
|------|--------|
| `--kv-cache-bytes-per-token` | **NÃO EXISTE** (passada pelo PS1 via `--server-arg` — inefetiva) |
| `--cache-type-k/v`, `--spec-type`, `--spec-draft-n-max`, `--ctx-size`, `--flash-attn`, `--no-warmup`, `--mmproj`, `--threads`, `--metrics` | OK |

### F5. PS1 (run-benchmark-auto.ps1, mtime 27/09 18:11)

- `--kv-cache-bytes-per-token 32768` → flag inexistente (F4); o cálculo de
  `estimated_context_bytes` usa fallback `args.ctx_size` (65536 × 32768 = 2 GB) —
  estimativa, não medida.
- `--mmproj "C:\...\benchmark_local\models\mmproj-F16.gguf"` → **arquivo inexistente**
  (`models/` tem só os 2 GGUFs de 11.77/12.04 GB). O real:
  `D:\models\ISTA-DASLab\Qwen3.8-27B-GSQ-RCO-GGUF\mmproj-Qwen3.8-27B-BF16.gguf`
  (GSQ) e `D:\models\27B unsloth\unsloth\Qwen3.8-27B-GGUF\mmproj-F16.gguf` (UD).
  `start_server` verifica `mmproj.exists()` e **silenciosamente não adiciona** —
  sem crash, mas sem aviso. Benchmark é de texto (prompts.json) → mmproj
  desnecessário; com 2 modelos de 12 GB em 16 GB VRAM, 0.93 GB de mmproj aperta.
- Preferência build CUDA local (`build-cuda\bin`) → **não existe mais** → cai no
  LLAMA_SERVER_EXE (winget Vulkan b11193). Consistente com o log do usuário.
- `--ctx-size 65536` → OK (build aceita; sweet spot medido 12/09).

### F6. Build Vulkan winget b11193
`--list-devices` → `Vulkan0: NVIDIA GeForce RTX 5060 Ti (16050 MiB, 15282 MiB free)`.
⚠️ NÃO é build CUDA — arquiteturas exóticas (TQ1_0/dspark) podem não carregar
(já documentado: ggml type 143 no b11193 Vulkan).

### F7. Suite de testes
65 testes em `scripts/test_benchmark.py` — **NENHUM cobre** `_post_json`,
`wait_for_server_ready`, timeout ou urlopen (grep sem matches). A falha exata que
quebrou a run não tem teste de regressão.

### F8. Limpeza git (baixa prioridade)
Arquivo untracked com nome corrompido:
`C\357\200\272UsersJasonDesktopPROJETOS...PLANO_ATUALIZACAO_MTP.md`
(`\357\200\272` = U+F03A, Private Use Area — bug de encoding chr(0xF03A) em script).
Remover o duplicado corrompido; manter `PLANO_ATUALIZACAO_MTP.md` correto.

---

## 3. Plano de correção

### A. benchmark_models.py (prioridade ALTA)

**A1. `_post_json` — capturar timeout de leitura (linha ~283):**

```python
    except urllib.error.HTTPError as exc:
        ...  # (mantém como está)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        elapsed = time.perf_counter() - start
        return 1, "", f"{type(exc).__name__}: {exc}", elapsed
```

`TimeoutError` é subclasse de `OSError` — capturar `OSError` cobre conexão e leitura.

**A2. `wait_for_server_ready` — probe robusto (linha ~526):**

```python
        remaining = max(5.0, deadline - time.time())
        status, body, error, _elapsed = _post_json(probe_url, {...}, timeout=min(60, remaining))
        if status == 200:
            return
        if status == 1 and "timeout" in error.lower():
            # 1º probe pós-load pode demorar >5s (compilação de kernels no Vulkan)
            time.sleep(2)
            continue
```

O deadline global de `args.server_start_timeout` (600s no PS1) continua controlando
o total — se o servidor nunca ficar pronto, `TimeoutError` final é levantado corretamente.

**A3. `model_has_mtp()` + header GGUF — corrigir falso negativo (linha ~423):**

```python
def gguf_has_mtp_header(model_path: Path) -> bool:
    """Detecta MTP embutido pela metadata do GGUF (primeiros 256 KB — não lê o arquivo todo)."""
    try:
        with open(model_path, "rb") as f:
            head = f.read(262144)
        return b"nextn_predict_layers" in head or b"mtp_num_hidden_layers" in head
    except OSError:
        return False
```

E em `resolve_spec_type`:
```python
    if requested == "auto":
        return ("draft-mtp"
                if model_has_mtp(model_path) or gguf_has_mtp_header(model_path)
                else "none")
```

Print do main (linha ~1106) já mostra o resultado — passa a ativar draft-mtp no
UD-IQ3_S automaticamente → consistente com o baseline de 12/09.

**A4. Print enganoso "servidor finalizado" (linha ~1221):**
Trocar por `"  servidor finalizado (cleanup)"` — deixa claro que foi o benchmark que
derrubou, não crash do modelo.

### B. run-benchmark-auto.ps1 (prioridade MÉDIA)

**B1.** Remover `--kv-cache-bytes-per-token 32768` (flag inexistente no b11193).

**B2.** Remover `--mmproj` hardcoded (benchmark de texto; alivia VRAM) **OU** apontar
para o real com checagem visível. Recomendado: remover — e se quiser benchmark de
visão no futuro, criar script separado com mmproj correto por família.

**B3.** Salvar com **UTF-8 com BOM** (regra do projeto — PowerShell 5.1 lê como ANSI).

### C. Teste de regressão (prioridade MÉDIA — scripts/test_benchmark.py)

**C1.** `_post_json` contra servidor que aceita conexão mas não responde (socket
server de mock) → retorna `(1, "", "TimeoutError: ...", elapsed)` em vez de levantar.

**C2.** `wait_for_server_ready` com 1º probe estourando timeout e servidor subindo
depois → retry até ficar pronto (não crash); deadline estourado → `TimeoutError`.

**C3.** `gguf_has_mtp_header` com GGUF fake contendo `nextn_predict_layers` → True;
sem → False.

**C4.** `resolve_spec_type("auto", <UD-IQ3_S com header MTP>)` → `draft-mtp`
(o caso real que falhou — regressão do F-B).

### D. Processo / limpeza (prioridade BAIXA)

**D1.** Remover arquivo untracked corrompido (F8):
`git clean -f "C\357\200\272UsersJasonDesktopPROJETOS03 - DONEbenchmark_localPLANO_ATUALIZACAO_MTP.md"`

**D2.** Após implementar A1-A4 + B1-B2: rodar suite (`python -m pytest scripts/test_benchmark.py`)
e re-executar `run-benchmark-auto.ps1` para validar end-to-end — verificar que o
UD-IQ3_S sobe com `spec detectado: draft-mtp (modelo com MTP embutido)`.

---

## 4. Critérios de aceite

- [ ] `_post_json` nunca deixa TimeoutError/OSError escapar (teste C1 verde)
- [ ] `wait_for_server_ready` faz retry em timeout de leitura (teste C2 verde)
- [ ] UD-IQ3_S detecta draft-mtp via header (teste C4 verde)
- [ ] PS1 sem `--kv-cache-bytes-per-token` e sem mmproj inexistente; UTF-8 com BOM
- [ ] Run completa end-to-end: 2 modelos testados, CSV + dashboard.json salvos
- [ ] Métrica spec_active/spec_draft_type no CSV coerente com o log do servidor
