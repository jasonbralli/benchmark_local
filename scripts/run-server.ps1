param(
    [string]$ModelPath
)

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot

# Try dynamic discovery: environment variable, PATH, then fallback
$SERVER_EXE = $null

if ($env:LLAMA_SERVER_EXE) {
    $SERVER_EXE = $env:LLAMA_SERVER_EXE
    Write-Host "✓ llama-server encontrado em LLAMA_SERVER_EXE: $SERVER_EXE" -ForegroundColor Green
}

if (-not $SERVER_EXE) {
    $found = Get-Command llama-server -ErrorAction SilentlyContinue
    if ($found) {
        $SERVER_EXE = $found.Source
        Write-Host "✓ llama-server encontrado no PATH: $SERVER_EXE" -ForegroundColor Green
    }
}

if (-not $SERVER_EXE) {
    $common = "C:\Users\$env:USERNAME\llama.cpp\build-cuda\bin\llama-server.exe"
    if (Test-Path $common) {
        $SERVER_EXE = $common
        Write-Host "✓ llama-server encontrado em localização comum: $SERVER_EXE" -ForegroundColor Green
    }
}

if (-not $SERVER_EXE) {
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

if (-not (Test-Path $SERVER_EXE)) {
    Write-Host "❌ Arquivo não existe: $SERVER_EXE" -ForegroundColor Red
    exit 1
}
$MODELS_DIR = Join-Path $ROOT "models"

if (-not $ModelPath) {
    $firstModel = Get-ChildItem -Path $MODELS_DIR -Filter *.gguf -File |
        Sort-Object Name |
        Select-Object -First 1

    if (-not $firstModel) {
        throw "Nenhum arquivo .gguf foi encontrado em $MODELS_DIR."
    }

    $ModelPath = $firstModel.FullName
}

if (-not (Test-Path -LiteralPath $ModelPath)) {
    throw "Modelo não encontrado: $ModelPath"
}

Write-Host "=== Iniciando llama-server com CUDA ===" -ForegroundColor Cyan
Write-Host "Modelo: $ModelPath" -ForegroundColor White
Write-Host "Servidor: http://localhost:8080" -ForegroundColor Green
Write-Host ""

& $SERVER_EXE `
    -m $ModelPath `
    --host localhost --port 8080 `
    -c 65536 `
    -ngl 99 `
    --threads 8 --threads-batch 8 `
    --batch-size 2048 `
    --ubatch-size 2048 `
    --flash-attn on `
    --cache-type-k q4_0 `
    --cache-type-v q4_0 `
    --temperature 0.6 `
    --top_p 0.95 --top_k 20 `
    --repeat_penalty 1.1 `
    -np 4 `
    --min-p 0.0 `
    -lv 2