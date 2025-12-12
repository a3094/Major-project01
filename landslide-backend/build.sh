#!/bin/bash
# Build script for Render deployment
# This ensures pip is upgraded and dependencies install correctly

set -e

echo "Checking Python version..."
python --version

echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "Installing requirements with pre-built wheels only..."
pip install --only-binary :all: -r requirements.txt || pip install -r requirements.txt

echo "Build complete!"

