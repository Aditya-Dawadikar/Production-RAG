#!/bin/bash
# Stage 1/3: Ubuntu packages, Java (for Elasticsearch) and the Python environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/_lib.sh"

log_stage "STAGE 1/3: Ubuntu packages, Java & Python environment"

log_info "Updating apt package lists..."
sudo apt-get update

log_info "Upgrading installed packages..."
sudo apt-get upgrade -y

log_info "Installing base packages (Java, Python, build tools)..."
sudo apt-get install -y \
    openjdk-21-jdk \
    python3 \
    python3-venv \
    python3-pip \
    build-essential \
    curl \
    wget \
    gnupg

log_success "System packages installed"

if [ ! -d "$PROJECT_ROOT/venv" ]; then
    log_info "Creating Python virtual environment at $PROJECT_ROOT/venv"
    python3 -m venv "$PROJECT_ROOT/venv"
    log_success "Virtual environment created"
else
    log_info "Virtual environment already exists, skipping creation"
fi

source "$PROJECT_ROOT/venv/bin/activate"

log_info "Upgrading pip..."
pip install --upgrade pip

log_info "Installing Python dependencies from requirements.txt..."
pip install -r "$PROJECT_ROOT/requirements.txt"

log_success "Python environment ready"
