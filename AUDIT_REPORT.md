# Relatório de Auditoria — benchmark_local

**Data:** 2026-07-04
**Scope:** `scripts/benchmark_models.py`, `scripts/benchmark_scorer.py`, `scripts/rescore.py`, `scripts/prompts.json`, `index.html`, `run-benchmark-auto.ps1`, `setup.ps1`, `scripts/run-server.ps1`

---

## Fase 1 — Bugs Críticos (corrigir imediatamente)

### BUG-01: `InstructionScorer.score()` — assinatura errada, benchmark quebra para prompts de instrução
**Arquivo:** `scripts/benchmark_scorer.py`, linha 620
**Severidade:** 🔴 CRÍTICO

A assinatura de `InstructionScorer.score()` no código atual é:
```python
def score(cls, response: str, expected_answer: Optional[str] = None) -> Tuple[float, Dict]:
```

Mas `evaluate_response()` na linha 679 chama assim:
```python
scorer.score(response, constraints, expected_elements)
```

São **3 argumentos posicionais** passados para um método que só aceita `response` + `expected_answer`. Isso gera um `TypeError: too many positional arguments` todo vez que um prompt da categoria `instruction` é avaliado.

Comparando com o backup (`benchmark_scorer.py.bak`), a assinatura original correta era:
```python
def score(cls, response: str, constraints: Optional[List[Dict]] = None) -> Tuple[float, Dict]:
```

**O que aconteceu:** A assinatura do `InstructionScorer` foi substituída pela do `ReasoningScorer` em alguma refatoração — provavelmente um copy-paste ou merge incorreto.

**Impacto:** Toda avaliação de prompts de instrução falha silenciosamente ou com erro, retornando scores zerados ou incorretos. O benchmark roda, mas os scores de `instruction` estão errados.

**Correção:** Restaurar a assinatura original:
```python
def score(cls, response: str, constraints: Optional[List[Dict]] = None) -> Tuple[float, Dict]:
```

---

### BUG-02: `sys.exc_info()` chamado fora de bloco `except`
**Arquivo:** `scripts/benchmark_scorer.py`, linha 84
**Severidade:** 🟡 MÉDIO

```python
# Sem code block ou parse direto falhou
lines_with_issues = len(str(sys.exc_info()[1]).split('\n'))
```

`sys.exc_info()` fora de um bloco `except` retorna `(None, None, None)`. O código acessa `[1]` → `None`, converte para string → `'None'`, split dá `['None']`, len = 1. **Sempre retorna 7.0** independentemente da gravidade do erro.

**Impacto:** Scores de syntax para respostas sem code block são artificialmente inflados para 7.0.

**Correção:** Salvar a exceção no primeiro `except SyntaxError` e reutilizá-la:
```python
except SyntaxError as e:
    # ... tenta extrair bloco ...
except Exception:
    return 0.0
# Fallback:
# Salvar a exceção original no primeiro except para reutilizar
```

---

### BUG-03: Inconsistência entre dashboard e scorer para classificação de avaliação
**Arquivo:** `index.html` (JS) vs `scripts/benchmark_scorer.py`
**Severidade:** 🟡 MÉDIO

O dashboard (JS) usa:
```javascript
if (score >= 8.5) return "Excelente";
if (score >= 7) return "Bom";
if (score >= 4) return "Regular";  // ← limiar 4
return "Fraco";
```

### BUG-06: `run-server.ps1` — caminho absoluto hardcoded ~~(FIXED)~~
**Status:** ✅ CORRIGIDO
**Arquivo:** `scripts/run-server.ps1`, linha 8

O hardcoded `$SERVER` na linha 73 foi substituído por `$SERVER_EXE`, que usa a lógica de descoberta dinâmica (variável de ambiente → PATH → fallback dinâmico com `$env:USERNAME`).

---

## Fase 2 — Bugs de Estabilidade (corrigir antes do próximo benchmark)

### BUG-04: `rescore.py` — import quebrado se executado de fora da pasta `scripts/` ~~(FIXED)~~
**Status:** ✅ CORRIGIDO — `sys.path.insert(0, str(Path(__file__).resolve().parent))` adicionado (linha 15).

---

### BUG-05: Contagem de tokens por palavras no modo CLI ~~(FIXED)~~
**Status:** ✅ CORRIGIDO — extrai `"decoded N tokens"` do stdout do `llama-cli` (linhas 903-908).

---

### BUG-07: Variável `log_metrics` calculada duas vezes ~~(FIXED)~~
**Status:** ✅ CORRIGIDO — cálculo único na linha 882.

---

### BUG-08: `run-benchmark-auto.ps1` — usa `python` sem verificar versão ~~(FIXED)~~
**Status:** ✅ CORRIGIDO — validação de versão Python 3+ antes de executar o benchmark.

---

## Fase 3 — Falhas de Qualidade no Código ~~(FIXED)~~
**Status:** ✅ TODAS CORRIDAS

### QUAL-01: `accuracy` no ExtractionScorer é proxy de `completeness` ~~(FIXED)~~
**Status:** ✅ Já estava corrigido — `score_accuracy` compara de fato com `expected_answer` (JSON field-by-field ou textual).

### QUAL-02: Thresholds hardcoded, sem configuração ~~(FIXED)~~
**Status:** ✅ CORRIGIDO — thresholds e pesos extraídos para `scripts/scoring_config.json`. Scorer carrega via `_load_config()` com fallback.

### QUAL-03: Heurísticas de scoring em português hardcoded ~~(FIXED)~~
**Status:** ✅ CORRIGIDO — padrões de linguagem agora vêm de `LANG_PATTERNS` no config. Suporte a `pt` e `en` configurado.

### QUAL-04: `CodingScorer.score_clarity` — variáveis detectadas por regex genérico ~~(FIXED)~~
**Status:** ✅ CORRIGIDO — regex busca snake_case (`[a-z][a-z0-9]_[a-z][a-z0-9]+`) E nomes longos (`[a-z]{5,}`).

### QUAL-05: `ReasoningScorer.score_complexity_handling` — heurística fraca ~~(FIXED)~~
**Status:** ✅ Já estava corrigido — conta identificadores únicos em vez de frequência de letras.

## Fase 4 — Segurança

### SEC-01: `run-server.ps1` — caminho hardcoded expõe estrutura do usuário ~~(FIXED)~~
**Status:** ✅ CORRIGIDO — variável `$SERVER` hardcoded removida; descoberta dinâmica via `$SERVER_EXE`.
**Arquivo:** `scripts/run-server.ps1`, linha 8

---

## Log de Mudanças — Fase 4 (Finalização)

**Data:** 2026-07-05

| ID | Descrição | Status | Arquivo |
|----|-----------|--------|---------|
| BUG-01 | InstructionScorer passa `expected_elements` → `topic_keywords` | ✅ Feito | `benchmark_scorer.py` |
| BUG-02 | `sys.exc_info()` fora de bloco `except` | ✅ Feito | `benchmark_scorer.py` |
| BUG-03 | Thresholds inconsistentes entre dashboard e config | ✅ Verificado | `index.html` / `scoring_config.json` |
| BUG-05 | Placeholder `...` (Ellipsis) em `clean_response()` | ✅ Removido | `benchmark_models.py` |
| BUG-06 | `$SERVER` hardcoded em `run-server.ps1` | ✅ Removido | `run-server.ps1` |
| BUG-07 | `log_metrics` calculado duas vezes | ✅ Corrigido | `benchmark_models.py` |
| BUG-08 | `python` sem verificar versão | ✅ Corrigido | `run-benchmark-auto.ps1` |
| QUAL-01 | `accuracy` como proxy de `completeness` | ✅ Corrigido | `benchmark_scorer.py` |
| QUAL-02 | Thresholds hardcoded sem configuração | ✅ Corrigido | `scoring_config.json` |
| QUAL-03 | Heurísticas hardcoded em português | ✅ Corrigido | `scoring_config.json` |
| QUAL-04 | Regex genérico para variáveis | ✅ Corrigido | `benchmark_scorer.py` |
| QUAL-05 | Heurística fraca em `complexity_handling` | ✅ Corrigido | `benchmark_scorer.py` |

### Testes

- **51 testes** criados em `scripts/test_benchmark.py`
- Cobertura: Config, CodingScorer, ExtractionScorer, InstructionScorer, ReasoningScorer, BenchmarkEvaluator, clean_response(), helpers, consistência de thresholds
- **51/51 passaram** (0 falhas)