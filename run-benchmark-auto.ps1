$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$MODELS = Join-Path $ROOT "models"
$OUT = Join-Path $ROOT "reports\benchmark_results.csv"
$BENCHMARK = Join-Path $ROOT "scripts\benchmark_models.py"

# Procura llama-server.exe de várias formas (por ordem de preferência):
# 1. Build CUDA local (mais compatível, suporta mais arquiteturas)
# 2. Variável de ambiente LLAMA_SERVER_EXE
# 3. No PATH

$SERVER = $null

# 1 — Build CUDA local (compilado manualmente, suporta mais arquiteturas como dspark)
$common = "C:\Users\$env:USERNAME\llama.cpp\build-cuda\bin\llama-server.exe"
$commonRelease = "C:\Users\$env:USERNAME\llama.cpp\build-cuda\bin\Release\llama-server.exe"
if (Test-Path $commonRelease) {
    $SERVER = $commonRelease
    Write-Host "✓ llama-server encontrado (build CUDA local Release): $SERVER" -ForegroundColor Green
} elseif (Test-Path $common) {
    $SERVER = $common
    Write-Host "✓ llama-server encontrado (build CUDA local): $SERVER" -ForegroundColor Green
}

# 2 — Variável de ambiente (sobrescreve apenas se o build local não existir)
if ((-not $SERVER) -and $env:LLAMA_SERVER_EXE) {
    $SERVER = $env:LLAMA_SERVER_EXE
    Write-Host "✓ llama-server encontrado em LLAMA_SERVER_EXE: $SERVER" -ForegroundColor Green
}

# 3 — No PATH
if (-not $SERVER) {
    $found = Get-Command llama-server -ErrorAction SilentlyContinue
    if ($found) {
        $SERVER = $found.Source
        Write-Host "✓ llama-server encontrado no PATH: $SERVER" -ForegroundColor Green
    }
}

if (-not $SERVER) {
    Write-Host "" -ForegroundColor Red
    Write-Host "❌ llama-server.exe não foi encontrado." -ForegroundColor Red
    Write-Host "" -ForegroundColor Red
    Write-Host "Solução: Defina a variável de ambiente LLAMA_SERVER_EXE:" -ForegroundColor Yellow
    Write-Host '    $env:LLAMA_SERVER_EXE = "C:\caminho\para\llama-server.exe"' -ForegroundColor Cyan
    Write-Host "" -ForegroundColor Red
    Write-Host "Ou adicione llama-server.exe ao PATH do Windows." -ForegroundColor Yellow
    Write-Host "" -ForegroundColor Red
    exit 1
}

if (-not (Test-Path $SERVER)) {
    Write-Host "❌ Arquivo não existe: $SERVER" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "📊 Benchmark Local de Modelos GGUF" -ForegroundColor Cyan
Write-Host "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" -ForegroundColor Cyan
Write-Host "Servidor: $SERVER" -ForegroundColor Gray
Write-Host "Modelos: $MODELS" -ForegroundColor Gray
Write-Host "Saída: $OUT" -ForegroundColor Gray
Write-Host ""

# Verify Python 3 is available
$pyVer = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "" -ForegroundColor Red
    Write-Host "❌ Python não foi encontrado. Instale Python 3 e tente novamente." -ForegroundColor Red
    exit 1
}
$major = [int]($pyVer -replace '[^0-9.]', '' -split '\.' | Select-Object -First 1)
if ($major -lt 3) {
    Write-Host "" -ForegroundColor Red
    Write-Host "❌ Python versão $pyVer detectada. Python 3+ é necessário." -ForegroundColor Red
    exit 1
}
Write-Host "Python: $pyVer" -ForegroundColor Gray

# Flags de performance repassadas DIRETO ao llama-server via --server-arg
# (start_server divide pares --flag value com shlex.split). Sem elas o server
# sobe com defaults do build (n_threads 8, 4 slots x ctx 4096, KV f16) ->
# offload parcial p/ CPU: GPU ~28%, tg ~14.7 t/s (auditoria 02/10).
python $BENCHMARK `
    $MODELS `
    --recursive `
    --server-exe $SERVER `
    --server-port 0 `
    --server-start-timeout 600 `
    --warmup-timeout 180 `
    --warmup-n-predict 12 `
    --warmup-prompt "Responda apenas com OK." `
    --ctx-size 65536 `
    --server-log-dir (Join-Path $ROOT "reports\server-logs") `
    --output $OUT `
    --n-predict 4096 `
    --temperature 0.6 `
    --top-p 0.95 `
    --repeat-penalty 1.1 `
    --seed 42 `
    --spec-type auto `
    --spec-draft-n-max 2 `
    --server-arg '-ngl 99' `
    --server-arg '--flash-attn on' `
    --server-arg '--cache-type-k q4_0' `
    --server-arg '--cache-type-v q4_0' `
    --server-arg '--parallel 1' `
    --server-arg '--ctx-size 65536' `
    --server-arg '--batch-size 2048' `
    --server-arg '--ubatch-size 1024' `
    --server-arg '--threads 14'

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "Benchmark concluido!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Resultados em:" -ForegroundColor Cyan
    Write-Host $OUT
    Write-Host ""
    Write-Host "Dashboard JSON:" -ForegroundColor Cyan
    Write-Host (Join-Path $ROOT "reports\benchmark_results.dashboard.json")
    Write-Host ""
    Write-Host "Para visualizar:" -ForegroundColor Yellow
    Write-Host "  1) Abra index.html e use o botao 'Carregar JSON'"
} else {
    Write-Host ""
    Write-Host "❌ Benchmark falhou com código $LASTEXITCODE" -ForegroundColor Red
    Write-Host ""
    exit $LASTEXITCODE
}
