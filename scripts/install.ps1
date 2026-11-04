# install ai-brainscan into .\.venv  (Windows PowerShell 5.1+)
#   scripts\install.ps1 [-Gpu] [-Extras web,tui,whitebox]
param(
  [switch]$Gpu,
  [string]$Extras = "web,tui"
)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONUTF8 = "1"

function Invoke-Checked {
  param([string]$Exe, [string[]]$Arguments)
  & $Exe @Arguments
  if ($LASTEXITCODE -ne 0) { throw "failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

$py = $null
$pre = @()
foreach ($cand in @("py -3", "python", "python3")) {
  $parts = $cand.Split(" ")
  $exe = $parts[0]
  $extra = @($parts | Select-Object -Skip 1)
  if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
  & $exe @extra -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>$null
  if ($LASTEXITCODE -eq 0) { $py = $exe; $pre = $extra; break }
}
if (-not $py) {
  Write-Host "need Python 3.10+ on PATH: https://www.python.org/downloads/"
  exit 1
}

$vpy = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Invoke-Checked $py ($pre + @("-m", "venv", ".venv")) }

Invoke-Checked $vpy @("-m", "pip", "install", "--upgrade", "pip")
if ($Gpu) {
  Invoke-Checked $vpy @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu128")
} else {
  Invoke-Checked $vpy @("-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu")
}
Invoke-Checked $vpy @("-m", "pip", "install", "-e", ".[$Extras]")

& $vpy -m brainscan doctor
Write-Host ""
Write-Host "next: scripts\setup.ps1   (creates .env, downloads the default model)"
Write-Host "then: scripts\run.ps1     (web UI)   |  scripts\run.ps1 tui  |  scripts\run.ps1 --help"
