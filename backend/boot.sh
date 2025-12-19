#!/bin/bash

# boot.sh - Fitvise Backend API Startup Script
# This script automates the process of starting the Fitvise backend server using uv
#
# Usage:
#   ./boot.sh        - Start server (skip dependency installation)
#   ./boot.sh -i     - Install dependencies and start server
#   ./boot.sh -h     - Show help

set -e  # Exit on any error

# Default values
SYNC_REQUIRED=false

# Function to show usage
show_help() {
    echo "🏋️ starting fitvise server"
    echo "======================================"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -i, --install    Install/update dependencies before starting"
    echo "  -h, --help       Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0               Start server (skip dependency installation)"
    echo "  $0 -i            Install dependencies and start server"
    echo ""
}

# Check if uv is installed
check_uv() {
    if ! command -v uv &> /dev/null; then
        echo "❌ uv is not installed. Please install it first:"
        echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
        echo "   # Or via pip: pip install uv"
        echo "   # Or via brew: brew install uv"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--install)
            SYNC_REQUIRED=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "🏋️ starting fitvise server"
echo "====================================="

# Check if uv is available
check_uv

# Step 1: Ensure virtual environment exists or create it
if [ ! -d ".venv" ]; then
    echo "📦 creating virtual environment with uv..."
    uv venv
    echo "✅ virtual environment created"
fi

# Step 2: Install/Update dependencies (conditional)
if [ "$SYNC_REQUIRED" = true ]; then
    echo "📥 installing or updating dependencies with uv..."
    uv sync
    echo "✅ dependencies installed/updated"
else
    echo "⏭️  skipping dependency installation (use -i to install)"
fi

# Step 3: Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  warning: .env file not found. you may need to create one."
    echo "   you can test configuration with: python test_settings.py"
fi

# Step 4: Start the server
echo "🚀 starting fitvise server..."
echo "   press Ctrl+C to stop the server"
echo ""

# Load environment variables and start the server
export $(grep -v '^#' .env | xargs) 2>/dev/null || true
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
uv run python run.py
