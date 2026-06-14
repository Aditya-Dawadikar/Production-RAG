#!/bin/bash
# Entry point: provisions a fresh EC2/Ubuntu instance to run the prod-rag
# FastAPI service end to end.
#
#   1. setup_ubuntu.sh       - apt packages, Java, Python venv & dependencies
#   2. setup_elasticsearch.sh - install Elasticsearch & populate the index
#   3. setup_chromadb.sh      - restore the Chroma vector store from S3
#
# Usage: bash setup/setup_ec2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

source "$SCRIPT_DIR/_lib.sh"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

log_banner "PROD-RAG EC2 SETUP"

bash "$SCRIPT_DIR/setup_ubuntu.sh"
bash "$SCRIPT_DIR/setup_elasticsearch.sh"
bash "$SCRIPT_DIR/setup_chromadb.sh"

log_banner "SETUP COMPLETE"

log_success "Everything is ready. Start the API with:"
echo ""
echo -e "    ${BOLD}source venv/bin/activate${NC}"
echo -e "    ${BOLD}uvicorn src.main:app --host 0.0.0.0 --port 8000${NC}"
echo ""
