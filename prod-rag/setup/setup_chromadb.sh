#!/bin/bash
# Stage 3/3: Restore the local Chroma vector store from its S3 backup.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/_lib.sh"

log_stage "STAGE 3/3: ChromaDB restore"

source "$PROJECT_ROOT/venv/bin/activate"

log_info "Restoring Chroma DB from S3 backup (if needed)..."
python "$SCRIPT_DIR/restore_chroma_from_s3.py"

log_success "ChromaDB ready"
