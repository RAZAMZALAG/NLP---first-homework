#!/usr/bin/env bash
set -e

# Install uv if not already present
if ! command -v uv &>/dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating virtual environment and installing dependencies..."
uv sync

echo ""
echo "Setup complete. Activate the environment with:"
echo "  source .venv/bin/activate"
