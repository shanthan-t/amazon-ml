param([Parameter(Mandatory=$true)][string]$RunDir)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
& (Join-Path $Repo ".venv\Scripts\python.exe") -m src.validate --run-dir $RunDir
if ($LASTEXITCODE -ne 0) { throw "Output validation failed." }
