param(
    [Parameter(Mandatory=$true)][string]$Source1,
    [string]$Output = "runs\benchmark_v1",
    [string]$Workers = "2,4,8,12,16",
    [int]$SampleRows = 4096,
    [int]$BudgetMinutes = 12
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
& (Join-Path $Repo ".venv\Scripts\python.exe") -m src.benchmark --source1 (Resolve-Path $Source1).Path --output $Output --workers $Workers --sample-rows $SampleRows --budget-minutes $BudgetMinutes
if ($LASTEXITCODE -ne 0) { throw "Benchmark failed." }
