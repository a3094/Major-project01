#!/bin/bash
# Build script for Render deployment
# This ensures pip is upgraded and dependencies install correctly

set -e

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
pip install -r requirements.txt

echo "Build complete!"

