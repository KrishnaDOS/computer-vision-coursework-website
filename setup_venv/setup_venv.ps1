#!/usr/bin/env pwsh
<#
Creates and activates a Python virtual environment and installs dependencies.
Usage: .\setup_venv.ps1 [-VenvDir <dir>] [-NoActivate] [-Recreate]
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

# --- STEP 1: PRE-CHECKS ---
Write-Host "--- Step 1: Initialization ---"
$Python = $env:PYTHON; if (-not $Python) { $Python = 'python' }
Write-Host "Using System Python interpreter: $Python"

if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
    Write-ErrAndExit "Error: $Python not found in PATH." 2
}

# --- STEP 2: CLEANUP ---
Write-Host "--- Step 2: Checking existing venvs ---"
$ExtraVenvs = @()
if ($VenvDir -ne '.venv' -and (Test-Path '.venv')) { $ExtraVenvs += '.venv' }
if ($VenvDir -ne 'venv' -and (Test-Path 'venv')) { $ExtraVenvs += 'venv' }

if ((Test-Path $VenvDir) -or ($ExtraVenvs.Count -gt 0)) {
    if ($Recreate.IsPresent) { $Confirm = $true } else {
        # Defaulting to False here to speed up re-runs, change to prompt if needed
        $Confirm = $false 
        Write-Host "Existing venv found. Reusing it (use -Recreate to force delete)."
    }

    if ($Confirm) {
        Write-Host "Removing old environments..."
        if (Test-Path $VenvDir) { Remove-Item -Recurse -Force -LiteralPath $VenvDir }
        foreach ($ev in $ExtraVenvs) { if (Test-Path $ev) { Remove-Item -Recurse -Force -LiteralPath $ev } }
    }
}

# --- STEP 3: CREATION ---
Write-Host "--- Step 3: Creating Venv ---"
if (-not (Test-Path $VenvDir)) {
    Write-Host "Creating virtual environment in $VenvDir..."
    try {
        & $Python -m venv $VenvDir
    } catch {
        Write-ErrAndExit "Error: Failed to create venv." 2
    }
} else {
    Write-Host "Virtualenv folder exists."
}

# --- STEP 4: ACTIVATION ---
Write-Host "--- Step 4: Activation ---"
$ActivateScript = Join-Path $VenvDir 'Scripts\Activate.ps1'
if (-not (Test-Path $ActivateScript)) {
    Write-Warning "CRITICAL: Activate script not found at $ActivateScript"
} elseif (-not $NoActivate.IsPresent) {
    try {
        Write-Host "Activating: $ActivateScript"
        . $ActivateScript
    } catch {
        Write-Warning "Activation script failed to run."
    }
}

# --- STEP 5: PIP UPGRADE (Protected) ---
Write-Host "--- Step 5: Upgrading Pip ---"

# Determine path to python executable inside venv
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'

if (-not (Test-Path $VenvPython)) {
    Write-Error "CRITICAL: Venv Python executable not found at: $VenvPython"
    Write-Host "Skipping Pip Upgrade..."
} else {
    try {
        Write-Host "Running pip upgrade using: $VenvPython"
        & $VenvPython -m pip install --upgrade pip setuptools wheel
    } catch {
        Write-Error "Pip upgrade failed. Continuing to requirements..."
    }
}

# --- STEP 6: REQUIREMENTS ---
Write-Host "--- Step 6: Requirements ---"

# Logic to find where this script is running from
$ScriptDir = $PSScriptRoot
if (-not $ScriptDir) { $ScriptDir = Get-Location }

$ReqInScript = Join-Path $ScriptDir 'requirements.txt'
$ReqInCwd = Join-Path (Get-Location) 'requirements.txt'
$ReqFile = $null

Write-Host "Searching for requirements.txt..."
Write-Host "  [?] Checked: $ReqInScript"
Write-Host "  [?] Checked: $ReqInCwd"

if (Test-Path $ReqInScript) { $ReqFile = $ReqInScript }
elseif (Test-Path $ReqInCwd) { $ReqFile = $ReqInCwd }

if ($ReqFile) {
    Write-Host "SUCCESS: Found requirements file at: $ReqFile"
    
    if (Test-Path $VenvPython) {
        Write-Host "Installing dependencies..."
        & $VenvPython -m pip install -r $ReqFile
    } else {
        Write-Error "Cannot install packages: Python executable missing."
    }
} else {
    Write-Warning "No requirements.txt found."
}

Write-Host "--- Setup Complete ---"