param(
    [Parameter(Mandatory=$true)][string]$Source1,
    [string]$RunDir = "runs\full_v6",
    [int]$Workers = 8,
    [int]$ThreadsPerWorker = 1
)
& (Join-Path $PSScriptRoot "run_windows.ps1") -Source1 $Source1 -RunDir $RunDir -Workers $Workers -ThreadsPerWorker $ThreadsPerWorker -Resume
