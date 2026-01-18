#!/bin/bash
#
# DEPA Training Demo - Modular GUI Edition
# ================================================
# Launches the dynamic web UI that auto-discovers any training scenario
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Banner
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                                                                  ║"
echo "║    ██████╗ ███████╗██████╗  █████╗     ████████╗██████╗         ║"
echo "║    ██╔══██╗██╔════╝██╔══██╗██╔══██╗    ╚══██╔══╝██╔══██╗        ║"
echo "║    ██║  ██║█████╗  ██████╔╝███████║       ██║   ██████╔╝        ║"
echo "║    ██║  ██║██╔══╝  ██╔═══╝ ██╔══██║       ██║   ██╔══██╗        ║"
echo "║    ██████╔╝███████╗██║     ██║  ██║       ██║   ██║  ██║        ║"
echo "║    ╚═════╝ ╚══════╝╚═╝     ╚═╝  ╚═╝       ╚═╝   ╚═╝  ╚═╝        ║"
echo "║                                                                  ║"
echo "║              Dynamic Scenario Runner - Modular GUI Edition       ║"
echo "║                                                                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export REPO_ROOT="$SCRIPT_DIR"
export TOOLS_HOME="$REPO_ROOT/external/confidential-sidecar-containers/tools"
DEMO_UI_DIR="$REPO_ROOT/gui-demo"
PORT=${PORT:-5001}

echo -e "${CYAN}[*]${NC} Repository: $REPO_ROOT"
echo ""

# Install prerequisites
command_exists() { command -v "$1" >/dev/null 2>&1; }

echo -e "${CYAN}[*]${NC} Installing prerequisites..."
./install-prerequisites.sh

# Setup Python environment
echo -e "${CYAN}[*]${NC} Setting up Python environment..."
cd "$DEMO_UI_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt 2>/dev/null
echo -e "${GREEN}[✓]${NC} Python environment ready"
echo ""

# Find available port
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}[!]${NC} Port $PORT in use, finding alternative..."
    for p in 5002 5003 5004 8080 8000 3000; do
        if ! lsof -Pi :$p -sTCP:LISTEN -t >/dev/null 2>&1; then
            PORT=$p
            break
        fi
    done
fi

# Get local IP
LOCAL_IP=$(hostname -I 2>/dev/null | awk '{print $1}' || echo "localhost")

# Discover scenarios
echo -e "${CYAN}[*]${NC} Discovering scenarios..."
SCENARIO_COUNT=$(ls -d "$REPO_ROOT/scenarios"/*/ 2>/dev/null | wc -l)
echo -e "${GREEN}[✓]${NC} Found ${BOLD}$SCENARIO_COUNT${NC} scenarios"
echo ""

# Launch
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║${NC}                                                                  ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  ${BOLD}DEPA Training Demo - Modular GUI Edition${NC}                        ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}                                                                  ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  ${CYAN}Local:${NC}    http://localhost:${PORT}                              ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  ${CYAN}Network:${NC}  http://${LOCAL_IP}:${PORT}                              ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}                                                                  ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  ${BOLD}Features:${NC}                                                      ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}    • Auto-discovers all scenarios                                ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}    • Parses export-variables.sh dynamically                      ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}    • Works with any new scenario without code changes            ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}                                                                  ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}  Press ${YELLOW}Ctrl+C${NC} to stop                                           ${GREEN}║${NC}"
echo -e "${GREEN}║${NC}                                                                  ${GREEN}║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Open browser
sleep 2
if command_exists xdg-open; then
    xdg-open "http://localhost:$PORT" 2>/dev/null &
elif command_exists open; then
    open "http://localhost:$PORT" 2>/dev/null &
fi

# Start server
python3 app.py $PORT

