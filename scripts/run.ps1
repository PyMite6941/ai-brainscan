# run brainscan from the venv. no args = web UI.
#   scripts\run.ps1            web UI on http://127.0.0.1:7860
#   scripts\run.ps1 tui        terminal UI
#   scripts\run.ps1 capture "prompt" --label refused
Set-Location (Split-Path -Parent $PSScriptRoot)
$vpy = Join-Path (Get-Location) ".venv\Scripts\python.exe"
if (-not (Test-Path $vpy)) { Write-Host "run scripts\install.ps1 first"; exit 1 }
$env:PYTHONUTF8 = "1"
$rest = @($args)
if ($rest.Count -eq 0) { $rest = @("web") }
& $vpy -m brainscan @rest
exit $LASTEXITCODE
