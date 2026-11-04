# first-run setup: .env, default model download, doctor. args pass through to `brainscan setup`.
Set-Location (Split-Path -Parent $PSScriptRoot)
$vpy = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Write-Host "run scripts\install.ps1 first"; exit 1 }
$env:PYTHONUTF8 = "1"
& $vpy -m brainscan setup @args
exit $LASTEXITCODE
