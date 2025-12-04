#!/usr/bin/env bash
set -euo pipefail

# Creates and activates a Python virtual environment and installs dependencies.
# Usage: ./scripts/setup_venv.sh [VENV_DIR] [--no-activate] [--recreate|-r]
# Default VENV_DIR is ./.venv

DEFAULT_VENV=".venv"
VENV_DIR=""
NO_ACTIVATE=false
RECREATE=false

# Parse arguments (positional VENV_DIR + optional flags)
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-activate)
      NO_ACTIVATE=true
      shift
      ;;
    --recreate|-r)
      RECREATE=true
      shift
      ;;
    -*|--*)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
    *)
      POSITIONAL+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL[@]} -gt 0 ]]; then
  VENV_DIR="${POSITIONAL[0]}"
else
  VENV_DIR="$DEFAULT_VENV"
fi

echo "Using Python interpreter: ${PYTHON:-python3}"

# prefer python3 from PATH
PYTHON=${PYTHON:-python3}

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Error: $PYTHON not found in PATH. Install Python 3 or set PYTHON env var to a valid interpreter." >&2
  exit 2
fi

# Detect other common virtualenv names to consider for deletion/recreation
EXTRA_VENVS=()
if [[ "$VENV_DIR" != ".venv" && -d ".venv" ]]; then
  EXTRA_VENVS+=(".venv")
fi
if [[ "$VENV_DIR" != "venv" && -d "venv" ]]; then
  EXTRA_VENVS+=("venv")
fi

if [[ -d "$VENV_DIR" || ${#EXTRA_VENVS[@]} -gt 0 ]]; then
  echo "Found existing virtualenv(s):"
  [[ -d "$VENV_DIR" ]] && echo "  - $VENV_DIR"
  for ev in "${EXTRA_VENVS[@]}"; do
    echo "  - $ev"
  done

  # If user asked to recreate or explicitly set --recreate, remove them
  if [[ "$RECREATE" == "true" ]]; then
    CONFIRM=true
  else
    read -r -p "Remove the existing virtualenv(s) above and recreate? [y/N] " resp
    case "$resp" in
      [yY]|[yY][eE][sS]) CONFIRM=true ;;
      *) CONFIRM=false ;;
    esac
  fi

  if [[ "$CONFIRM" == "true" ]]; then
    # If a virtualenv is currently active, try to deactivate it first
    if [[ -n "${VIRTUAL_ENV-}" ]]; then
      echo "Active virtualenv detected at ${VIRTUAL_ENV}. Attempting to deactivate..."
      if type deactivate >/dev/null 2>&1; then
        # shellcheck disable=SC2034
        deactivate || true
      else
        echo "Warning: 'deactivate' function not available in this shell. Proceeding to remove folders anyway." >&2
      fi
    fi

    # Remove all candidates
    if [[ -d "$VENV_DIR" ]]; then
      echo "Removing $VENV_DIR"
      rm -rf -- "$VENV_DIR"
    fi
    for ev in "${EXTRA_VENVS[@]}"; do
      if [[ -d "$ev" ]]; then
        echo "Removing $ev"
        rm -rf -- "$ev"
      fi
    done
  else
    echo "Keeping existing virtualenv(s); will reuse $VENV_DIR if present."
  fi
fi

if [[ -d "$VENV_DIR" ]]; then
  echo "Virtualenv already exists at $VENV_DIR — reusing it."
else
  echo "Creating virtual environment in $VENV_DIR..."
  # check that venv and ensurepip are available in the selected interpreter
  if ! "$PYTHON" -c 'import venv, ensurepip' >/dev/null 2>&1; then
    echo "Error: your Python interpreter does not have the venv/ensurepip modules available." >&2
    echo "On Debian/Ubuntu install the venv support (example): sudo apt install python3-venv" >&2
    echo "On other distros, install the package that provides ensurepip/venv or use a Python build with ensurepip enabled." >&2
    exit 2
  fi

  "$PYTHON" -m venv "$VENV_DIR"
fi

ACTIVATE="$VENV_DIR/bin/activate"
if [[ ! -f "$ACTIVATE" ]]; then
  echo "Error: activate script missing at $ACTIVATE" >&2
  exit 3
fi

if [[ "$NO_ACTIVATE" == "false" ]]; then
  # shellcheck disable=SC1090
  source "$ACTIVATE"
  echo "Activated virtualenv at $VENV_DIR"
fi

echo "Upgrading pip and setuptools..."
"$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel

# Prefer a requirements.txt located next to this script, fall back to the
# current working directory. This avoids failures when the script is executed
# from the repo root but the requirements live in the script folder.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_IN_SCRIPT="$SCRIPT_DIR/requirements.txt"
REQ_IN_CWD="$PWD/requirements.txt"
REQ_FILE=""

if [[ -f "$REQ_IN_SCRIPT" ]]; then
  REQ_FILE="$REQ_IN_SCRIPT"
elif [[ -f "$REQ_IN_CWD" ]]; then
  REQ_FILE="$REQ_IN_CWD"
fi

if [[ -n "$REQ_FILE" ]]; then
  echo "Installing packages from $REQ_FILE"
  "$VENV_DIR/bin/pip" install -r "$REQ_FILE"
else
  echo "No requirements file found — skipping package installation."
  echo "Create a requirements.txt in the script or working directory to install packages automatically." >&2
fi

echo "Done. To activate the venv run: source $VENV_DIR/bin/activate"
