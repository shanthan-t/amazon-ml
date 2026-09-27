param(
    [Parameter(Mandatory=$true)][string]$Source1,
    [string]$RunDir = "runs\full_v6"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
& (Join-Path $Repo ".venv\Scripts\python.exe") -m src.merge --run-dir $RunDir --source1 (Resolve-Path $Source1).Path
if ($LASTEXITCODE -ne 0) { throw "Merge/validation failed." }
