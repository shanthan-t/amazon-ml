param([Parameter(Mandatory=$true)][string]$RunDir)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
& (Join-Path $Repo ".venv\Scripts\python.exe") -m src.progress --run-dir $RunDir
