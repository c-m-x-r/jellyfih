#!/usr/bin/env python
"""Check if all dependencies are installed for the web viewer."""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"✓ {package_name}")
        return True
    except ImportError:
        print(f"✗ {package_name} (missing)")
        return False

def main():
    print("Checking dependencies for Jellyfih Web Viewer...")
    print()

    deps = [
        ('flask', 'flask'),
        ('numpy', 'numpy'),
        ('matplotlib', 'matplotlib'),
        ('scipy', 'scipy'),
    ]

    missing = []
    for module, package in deps:
        if not check_import(module, package):
            missing.append(package)

    print()

    if missing:
        print(f"Missing {len(missing)} dependencies.")
        print()
        print("Install with:")
        print(f"  pip install {' '.join(missing)}")
        print()
        return 1
    else:
        print("All dependencies installed! ✓")
        print()
        print("Start the web viewer with:")
        print("  cd web/")
        print("  python app.py")
        print()
        return 0

if __name__ == '__main__':
    sys.exit(main())
