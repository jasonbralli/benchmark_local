$ErrorActionPreference = 'Stop'
$ROOT = $PSScriptRoot

Write-Host ''
Write-Host 'Setup: Benchmark Local de Modelos GGUF' -ForegroundColor Cyan
Write-Host '==========================================' -ForegroundColor Cyan
Write-Host ''

$python = Get-Command python -ErrorAction SilentlyContinue
if ($python) {
    & python --version
} else {
    Write-Host 'Python nao encontrado. Instale Python 3.9+' -ForegroundColor Red
    exit 1
}

$SERVER = $null
if ($env:LLAMA_SERVER_EXE) {
    $SERVER = $env:LLAMA_SERVER_EXE
}
if (-not $SERVER) {
    $found = Get-Command llama-server -ErrorAction SilentlyContinue
    if ($found) {
        $SERVER = $found.Source
    }
}
if (-not $SERVER) {
    $SERVER = Read-Host 'Caminho para llama-server.exe (deixe vazio para abortar)'
    if (-not $SERVER) { exit 1 }
}
if (-not (Test-Path $SERVER)) {
    Write-Host "Caminho invalido: $SERVER" -ForegroundColor Red
    exit 1
}
[Environment]::SetEnvironmentVariable('LLAMA_SERVER_EXE', $SERVER, 'Process')
try {
    [Environment]::SetEnvironmentVariable('LLAMA_SERVER_EXE', $SERVER, 'User')
} catch {}

$dirs = @('models','reports','reports/server-logs')
foreach ($dir in $dirs) {
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
}

$checks = @(
    'scripts/benchmark_models.py',
    'readme/README.md',
    'index.html',
    '.gitignore',
    'run-benchmark-auto.ps1'
)
foreach ($f in $checks) {
    if (Test-Path $f) {
        Write-Host "OK: $f" -ForegroundColor Green
    } else {
        Write-Host "FALTA: $f" -ForegroundColor Red
    }
}

Write-Host ''
Write-Host 'Setup finalizado.' -ForegroundColor Green
