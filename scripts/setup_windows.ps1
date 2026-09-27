param(
    [string]$Python = "3.12"
)
$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
Set-Location $Repo

$PyLauncher = Get-Command py -ErrorAction SilentlyContinue
if (-not $PyLauncher) { throw "Python Launcher 'py' is required. Install 64-bit CPython 3.12 and rerun." }
$PySelector = "-$Python"
$VersionText = & py $PySelector -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($LASTEXITCODE -ne 0 -or $VersionText.Trim() -ne $Python) {
    throw "CPython $Python is required. Install it, then run: .\scripts\setup_windows.ps1 -Python $Python"
}

if (-not (Test-Path ".venv\Scripts\python.exe")) { & py $PySelector -m venv .venv }
$Vpy = Join-Path $Repo ".venv\Scripts\python.exe"
& $Vpy -m pip install --upgrade pip
& $Vpy -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

# Obtain the official pinned ICU4C binary package if its required DLLs are absent.
$IcuDir = Join-Path $Repo "native\icu"
New-Item -ItemType Directory -Force -Path $IcuDir | Out-Null
$RequiredDlls = @("icudt77.dll", "icuuc77.dll", "icuin77.dll")
$MissingDlls = @($RequiredDlls | Where-Object { -not (Test-Path (Join-Path $IcuDir $_)) })
if ($MissingDlls.Count -gt 0) {
    $Zip = Join-Path $env:TEMP "icu4c-77_1-Win64-MSVC2022.zip"
    $Url = "https://github.com/unicode-org/icu/releases/download/release-77-1/icu4c-77_1-Win64-MSVC2022.zip"
    Write-Host "Downloading official ICU4C 77.1 Windows x64 binaries..."
    Invoke-WebRequest -Uri $Url -OutFile $Zip
    $ZipHash = (Get-FileHash -LiteralPath $Zip -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ZipHash -ne "6b62471ed2895959d6a85c64c58572ac677734547fe8383e6e3fd706a06dc3fa") {
        throw "ICU4C archive SHA-256 mismatch: $ZipHash"
    }
    $Extract = Join-Path $env:TEMP ("icu4c-77_1-" + [guid]::NewGuid().ToString("N"))
    Expand-Archive -LiteralPath $Zip -DestinationPath $Extract
    foreach ($Name in $RequiredDlls) {
        $Found = Get-ChildItem -Path $Extract -Filter $Name -Recurse | Select-Object -First 1
        if (-not $Found) { throw "Official ICU package did not contain $Name" }
        Copy-Item -LiteralPath $Found.FullName -Destination (Join-Path $IcuDir $Name) -Force
    }
    Remove-Item -LiteralPath $Extract -Recurse -Force
}

Write-Host "Checking Python packages, ICU version/transliteration, model hashes, and copied retrieval artifacts..."
& $Vpy -m src.setup_check
if ($LASTEXITCODE -ne 0) { throw "Setup integrity checks failed. Read the missing-artifact report above." }
Write-Host "Windows V6 environment is ready. Run parity before inference: .\.venv\Scripts\python.exe -m src.verify_parity"
