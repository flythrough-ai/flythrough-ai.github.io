#!/bin/bash
set -e  # Exit on any error

# Show current directory and Python version
echo "Current directory: $(pwd)"
echo "Python version: $(python --version)"

# Install uv for better dependency resolution
echo "Installing uv package manager..."
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="/opt/render/.cargo/bin:$PATH"

# Verify uv is installed
which uv || echo "uv not found in PATH"
uv --version || echo "uv command failed"

# Create data directories
echo "Creating persistent data directories..."
mkdir -p /opt/render/project/data/uploads
mkdir -p /opt/render/project/data/thumbs

# Install dependencies with uv or pip as fallback
echo "Installing dependencies..."
if command -v uv &> /dev/null; then
    echo "Using uv to install dependencies..."
    uv pip install --system -r requirements.txt || {
        echo "uv installation failed, falling back to pip..."
        pip install --no-cache-dir -r requirements.txt
    }
else
    echo "uv not available, using pip..."
    pip install --no-cache-dir -r requirements.txt
fi

# Initialize the database
echo "Initializing database..."
python update_db.py

echo "Setup complete!"