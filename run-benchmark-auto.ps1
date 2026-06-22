$ErrorActionPreference = "Stop"

$ROOT = $PSScriptRoot
$MODELS = Join-Path $ROOT "models"
$OUT = Join-Path $ROOT "reports\benchmark_results.csv"
$BENCHMARK = Join-Path $ROOT "scripts\benchmark_models.py"

# Procura llama-server.exe de várias formas:
# 1. Variável de ambiente LLAMA_SERVER_EXE
# 2. No PATH
# 3. Fallback local (comentado para não quebrar se não existir)

$SERVER = $null

if ($env:LLAMA_SERVER_EXE) {
    $SERVER = $env:LLAMA_SERVER_EXE
    Write-Host "✓ llama-server encontrado em LLAMA_SERVER_EXE: $SERVER" -ForegroundColor Green
}

if (-not $SERVER) {
    $found = Get-Command llama-server -ErrorAction SilentlyContinue
    if ($found) {
        $SERVER = $found.Source
        Write-Host "✓ llama-server encontrado no PATH: $SERVER" -ForegroundColor Green
    }
}

# Fallback para localização comum do llama.cpp (comentado — descomente se usar)
# if (-not $SERVER) {
#     $common = "C:\Users\$env:USERNAME\llama.cpp\build-cuda\bin\llama-server.exe"
#     if (Test-Path $common) {
#         $SERVER = $common
#         Write-Host "✓ llama-server encontrado em localização comum: $SERVER" -ForegroundColor Green
#     }
# }

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

python $BENCHMARK `
    $MODELS `
    --recursive `
    --server-exe $SERVER `
    --server-port 0 `
    --server-start-timeout 600 `
    --warmup-timeout 180 `
    --warmup-n-predict 12 `
    --warmup-prompt "Responda apenas com OK." `
    --ctx-size 64512 `
    --kv-cache-bytes-per-token 32768 `
    --server-log-dir (Join-Path $ROOT "reports\server-logs") `
    --output $OUT `
    --n-predict 128 `
    --temperature 0.6 `
    --top-p 0.95 `
    --repeat-penalty 1.1 `
    --seed 42

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
    Write-Host "  1) Abra index.html e use o botao 'Carregar JSON', ou"
    Write-Host "  2) Rode: python -m http.server 8000  |>  http://localhost:8000/"
} else {
    Write-Host ""
    Write-Host "❌ Benchmark falhou com código $LASTEXITCODE" -ForegroundColor Red
    Write-Host ""
    exit $LASTEXITCODE
}
