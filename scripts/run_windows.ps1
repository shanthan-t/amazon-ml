param(
    [Parameter(Mandatory=$true)][string]$Source1,
    [string]$RunDir = "runs\full_v6",
    [int]$Workers = 8,
    [int]$ThreadsPerWorker = 1,
    [switch]$Resume
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo
$ArgsList = @("-m", "windows_inference.runner", "--source1", (Resolve-Path $Source1).Path,
              "--run-dir", $RunDir, "--workers", "$Workers",
              "--threads-per-worker", "$ThreadsPerWorker")
if ($Resume) { $ArgsList += "--resume" }
& (Join-Path $Repo ".venv\Scripts\python.exe") @ArgsList
if ($LASTEXITCODE -ne 0) { throw "Inference failed; rerun with -Resume after checking worker logs." }
