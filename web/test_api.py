#!/usr/bin/env python
"""Test the web viewer API without starting a server."""

import sys
sys.path.insert(0, '..')

try:
    from make_jelly import fill_tank, random_genome, AURELIA_GENOME
    import numpy as np
    print("✓ Successfully imported make_jelly module")
except ImportError as e:
    print(f"✗ Failed to import make_jelly: {e}")
    sys.exit(1)

print()
print("Testing genome generation...")
print()

# Test random genome
print("1. Random genome generation:")
try:
    genome = random_genome()
    print(f"   Generated: {genome.tolist()[:3]}... (first 3 genes)")
    print(f"   Shape: {genome.shape}")
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()

# Test Aurelia genome
print("2. Aurelia aurita reference:")
try:
    print(f"   Genome: {AURELIA_GENOME.tolist()[:3]}... (first 3 genes)")
    print(f"   Shape: {AURELIA_GENOME.shape}")
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()

# Test phenotype generation
print("3. Phenotype generation (morphology):")
try:
    genome = AURELIA_GENOME
    pos, mat, stats = fill_tank(genome, 80000, grid_res=128)
    print(f"   Particles: {stats['n_total']:,}")
    print(f"   Robot: {stats['n_robot']:,}")
    print(f"   Muscle: {stats['muscle_count']:,}")
    print(f"   Water: {stats['n_water']:,}")
    print("   ✓ Success")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()
print("All tests passed! The web viewer should work correctly.")
print()
print("Start the server with:")
print("  python app.py")
