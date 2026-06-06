$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

function Find-Python310 {
    try {
        $version = & py -3.10 --version 2>$null
        if ($version -match "Python 3\.10") {
            return "py -3.10"
        }
    } catch {}

    foreach ($exe in @("python3.10", "python")) {
        try {
            $version = & $exe --version 2>$null
            if ($version -match "Python 3\.10") {
                return $exe
            }
        } catch {}
    }

    return ""
}

$PythonCmd = Find-Python310

if (-not $PythonCmd) {
    Write-Host "Python 3.10 was not found."
    Write-Host "Install it with:"
    Write-Host "  winget install --id Python.Python.3.10 -e"
    Write-Host "Then run .\scripts\setup_windows.ps1 again."
    exit 1
}

Write-Host "Using $PythonCmd"
Invoke-Expression "$PythonCmd -m venv .venv"

$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
& $Python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch CPU wheels. This is safest because no NVIDIA GPU was detected."
& $Python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
& $Python -m pip install -r requirements.txt

Write-Host ""
Write-Host "Environment ready:"
Write-Host "  .\.venv\Scripts\python.exe --version"
Write-Host "  .\scripts\run_smoke_test.ps1"
