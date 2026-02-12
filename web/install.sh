#!/bin/bash
# Simple pip-based installation for web viewer (no uv needed)

echo "🧬 Jellyfih Web Viewer - Installation"
echo ""

# Install dependencies directly with pip (works on Termux/Android)
echo "Installing dependencies with pip..."
pip install flask numpy scipy matplotlib pillow

echo ""
echo "✓ Installation complete!"
echo ""
echo "Test with:"
echo "  cd web/"
echo "  python test_api.py"
echo ""
echo "Run with:"
echo "  python app.py"
