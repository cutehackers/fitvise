#!/bin/bash

# boot.sh - Fitvise Backend API Startup Script
# This script automates the process of starting the Fitvise backend server
#
# Usage:
#   ./boot.sh        - Start server (skip dependency installation)
#   ./boot.sh -i     - Install dependencies and start server
#   ./boot.sh -h     - Show help

set -e  # Exit on any error

# Default values
INSTALL_DEPS=false

# Function to show usage
show_help() {
    echo "🏋️ Starting Fitvise server"
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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--install)
            INSTALL_DEPS=true
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

echo "🏋️ Starting Fitvise server"
echo "====================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please create one first:"
    echo "   python -m venv venv"
    exit 1
fi

# Step 1: Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Verify virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Step 2: Install/update dependencies (conditional)
if [ "$INSTALL_DEPS" = true ]; then
    echo "📥 Installing/updating dependencies..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed/updated"
else
    echo "⏭️  Skipping dependency installation (use -i to install)"
fi

# Step 3: Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found. You may need to create one."
    echo "   You can test configuration with: python test_settings.py"
fi

# Step 4: Test configuration (optional)
echo "🔧 Testing configuration..."
if python test_settings.py > /dev/null 2>&1; then
    echo "✅ Configuration test passed"
else
    echo "⚠️  Configuration test failed - check your .env file"
    echo "   Continue anyway? (y/N)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        echo "❌ Startup cancelled"
        exit 1
    fi
fi

# Step 5: Start the server
echo "🚀 Starting Fitvise Backend API server..."
echo "   Press Ctrl+C to stop the server"
echo ""

python run.py