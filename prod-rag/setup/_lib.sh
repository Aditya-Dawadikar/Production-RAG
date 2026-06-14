# Shared logging helpers for prod-rag setup scripts.
# Source this file from other setup scripts: source "$SCRIPT_DIR/_lib.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[1;36m'
BOLD='\033[1m'
NC='\033[0m'

log_banner() {
    echo -e "\n${BOLD}${BLUE}########################################################${NC}"
    echo -e "${BOLD}${BLUE}#  $1${NC}"
    echo -e "${BOLD}${BLUE}########################################################${NC}\n"
}

log_stage() {
    echo -e "\n\n${BLUE}========================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}========================================================${NC}\n"
}

log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}
