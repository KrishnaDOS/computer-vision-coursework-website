#!/usr/bin/env pwsh
<#
Creates and activates a Python virtual environment and installs dependencies.
Usage: .\setup_venv.ps1 [-VenvDir <dir>] [-NoActivate] [-Recreate]
Default VenvDir is ./.venv

Notes:
- To have the activation persist in your current session, dot-source this script:
  `. .\setup_venv.ps1` or run with an execution-policy bypass:
  `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force; . .\setup_venv.ps1`
#>

[CmdletBinding()]
param(
    [Parameter(Position=0)]
    [string]
    $VenvDir = ".venv",

    [switch]
    $NoActivate,

    [Alias('r')]
    [switch]
    $Recreate
)

function Write-ErrAndExit($msg, $code=1) {
    Write-Error $msg
    exit $code
}

$Python = $env:PYTHON; if (-not $Python) { $Python = 'python' }
Write-Host "Using Python interpreter: $Python"

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-ErrAndExit "Error: $Python not found in PATH. Install Python 3 or set PYTHON env var to a valid interpreter." 2
}

# Detect other common virtualenv names to consider for deletion/recreation
$ExtraVenvs = @()
if ($VenvDir -ne '.venv' -and (Test-Path '.venv')) { $ExtraVenvs += '.venv' }
if ($VenvDir -ne 'venv' -and (Test-Path 'venv')) { $ExtraVenvs += 'venv' }

if ((Test-Path $VenvDir) -or ($ExtraVenvs.Count -gt 0)) {
    Write-Host "Found existing virtualenv(s):"
    if (Test-Path $VenvDir) { Write-Host "  - $VenvDir" }
    foreach ($ev in $ExtraVenvs) { Write-Host "  - $ev" }

    if ($Recreate.IsPresent) { $Confirm = $true } else {
        $resp = Read-Host "Remove the existing virtualenv(s) above and recreate? [y/N]"
        if ($resp -match '^(?i:y|yes)$') { $Confirm = $true } else { $Confirm = $false }
    }

    if ($Confirm) {
        if ($env:VIRTUAL_ENV) {
            Write-Host "Active virtualenv detected at ${env:VIRTUAL_ENV}. Attempting to deactivate..."
            if (Get-Command deactivate -ErrorAction SilentlyContinue) {
                try { deactivate } catch { }
            } else {
                Write-Warning "'deactivate' function not available in this session. Proceeding to remove folders anyway."
            }
        }

        if (Test-Path $VenvDir) {
            Write-Host "Removing $VenvDir"
            Remove-Item -Recurse -Force -LiteralPath $VenvDir
        }
        foreach ($ev in $ExtraVenvs) {
            if (Test-Path $ev) {
                Write-Host "Removing $ev"
                Remove-Item -Recurse -Force -LiteralPath $ev
            }
        }
    } else {
        Write-Host "Keeping existing virtualenv(s); will reuse $VenvDir if present."
    }
}

if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment in $VenvDir..."
    try {
        & $Python -m venv $VenvDir
    } catch {
        Write-ErrAndExit "Error: your Python interpreter failed to create the venv. Ensure the interpreter has venv/ensurepip available." 2
    }
} else {
    Write-Host "Virtualenv already exists at $VenvDir — reusing it."
}

$Activate = Join-Path $VenvDir 'Scripts\Activate.ps1'
if (-not (Test-Path $Activate)) {
    Write-Warning "Activate script missing at $Activate"
}

if (-not $NoActivate.IsPresent) {
    try {
        if (Test-Path $Activate) {
            Write-Host "Attempting to activate virtualenv at $VenvDir"
            . $Activate
            Write-Host "Activated virtualenv at $VenvDir"
        } else {
            Write-Warning "Cannot find activation script. You can activate manually later: `. $VenvDir\Scripts\Activate.ps1`"
        }
    } catch {
        Write-Warning "Failed to activate automatically. To activate manually run: `. $VenvDir\Scripts\Activate.ps1`"
    }
}

Write-Host "Upgrading pip and setuptools..."
& (Join-Path $VenvDir 'Scripts\python.exe') -m pip install --upgrade pip setuptools wheel

# Prefer a requirements.txt located next to this script, fall back to the
# current working directory. This avoids failures when the script is executed
# from the repo root but the requirements live in the script folder.
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ReqInScript = Join-Path $ScriptDir 'requirements.txt'
$ReqInCwd = Join-Path (Get-Location) 'requirements.txt'
$ReqFile = $null

if (Test-Path $ReqInScript) { $ReqFile = $ReqInScript }
elseif (Test-Path $ReqInCwd) { $ReqFile = $ReqInCwd }

if ($ReqFile) {
    Write-Host "Installing packages from $ReqFile"
    & (Join-Path $VenvDir 'Scripts\pip.exe') install -r $ReqFile
} else {
    Write-Host "No requirements file found — skipping package installation."
    Write-Host "Create a requirements.txt in the script or working directory to install packages automatically."
}

Write-Host "Done. To activate the venv run (PowerShell): `. $VenvDir\Scripts\Activate.ps1`"
Write-Host "Or (cmd.exe): $VenvDir\Scripts\activate.bat"
