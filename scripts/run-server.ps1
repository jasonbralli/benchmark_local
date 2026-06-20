param(
    [string]$ModelPath
)

$ErrorActionPreference = "Stop"

$ROOT = Split-Path -Parent $PSScriptRoot
$SERVER = "C:\Users\Jason\llama.cpp\build-cuda\bin\llama-server.exe"
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

& $SERVER `
    -m $ModelPath `
    --host localhost --port 8080 `
    -c 64512 `
    -ngl 99 `
    --threads 14 --threads-batch 10 `
    --temperature 0.6 `
    --top_p 0.95 --top_k 20 `
    --repeat_penalty 1.1 `
    -fa on `
    --cache-type-k q8_0 `
    --cache-type-v q8_0 `
    -np 1 `
    --min-p 0.0 `
    -lv 2 `
