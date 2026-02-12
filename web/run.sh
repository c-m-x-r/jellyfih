#!/bin/bash
# Jellyfih Web Viewer startup script

echo "=== Jellyfih Web Viewer ==="
echo ""

# Check Python version
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check dependencies
echo "Checking dependencies..."
missing_deps=()

for dep in flask numpy matplotlib scipy; do
    if ! python -c "import $dep" 2>/dev/null; then
        missing_deps+=("$dep")
    fi
done

if [ ${#missing_deps[@]} -gt 0 ]; then
    echo ""
    echo "Missing dependencies: ${missing_deps[*]}"
    echo ""
    echo "Install with:"
    echo "  pip install ${missing_deps[*]}"
    echo ""
    exit 1
fi

echo "All dependencies installed ✓"
echo ""

# Start Flask
echo "Starting Flask server..."
echo "Access at: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

python app.py
