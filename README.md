# benchmark_local

Benchmarks de modelos locais GGUF em um painel unificado e autossuficiente.

## Descrição

Este projeto avalia automaticamente modelos locais de linguagem (formato GGUF) em quatro categorias de tarefas:
- **Coding** — geração e edição de código Python
- **Extraction** — extração de dados estruturados
- **Instruction** — conformidade com instruções e restrições
- **Reasoning** — raciocínio lógico e matemático

Fornece métricas de performance (tokens/s), qualidade (score automatizado), consistência (variância) e uso de recursos (memória).

## Requisitos

- Python 3.9+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) (llama-server.exe)
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
- Procura `llama-server.exe` (variável de ambiente, PATH, ou manual)
- Sobe um servidor por modelo (porta aleatória)
- Realiza warmup para estabilizar o modelo
- Executa 20 prompts distribuídos em 4 categorias (5 por categoria)
- Avalia as respostas com scoring automatizado
- Gera os relatórios

## Saídas geradas

```
reports/
├── benchmark_results.csv              # Registros brutos de todos os prompts
├── benchmark_results.dashboard.json   # Payload do dashboard (carregável por arrastar)
├── scoring_summary.json               # Resumo de avaliação por modelo
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
├── README.md                       # Documentação deste arquivo
├── LICENSE                         # Licença MIT
├── setup.ps1                       # Script de preparação do ambiente
├── run-benchmark-auto.ps1          # Script principal de benchmark (PS1)
├── index.html                      # Dashboard visual (carrega JSON via botão)
├── prompts.json                    # 20 prompts de avaliação (5 por categoria)
├── models/                         # Pasta para arquivos .gguf (não versionado)
├── reports/                        # Relatórios gerados (não versionado)
│   ├── benchmark_results.csv
│   ├── benchmark_results.dashboard.json
│   ├── scoring_summary.json
│   └── server-logs/
└── scripts/
    ├── benchmark_models.py         # Motor de benchmark (Python)
    ├── benchmark_scorer.py         # Sistema de avaliação automatizada
    └── run-server.ps1              # Script auxiliar para rodar servidor
```

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

## Melhorias Futuras

- [ ] Suporte a múltiplos modelos em paralelo
- [ ] Comparação visual com gráficos de barras
- [ ] Configuração de prompts customizados via JSON
- [ ] Suporte a outros formatos (GGML, AWQ, EXL2)
- [ ] Integração com GitHub Actions para CI
- [ ] Dashboard com filtros por categoria
- [ ] Exportação de relatórios em PDF com marcação automática
- [ ] Suporte a avaliação com LLM externo (feedback de modelo)
- [ ] Configuração via `.env` para variáveis de ambiente
- [ ] Suporte a Windows Terminal e WSL

## Licença

MIT License — veja o arquivo [LICENSE](LICENSE).
