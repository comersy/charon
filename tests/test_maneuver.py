"""Unit tests for core/maneuver.py."""
import sys
sys.path.insert(0, ".")

import numpy as np
from core.maneuver import hohmann_transfer, lambert_transfer, MU_EARTH


def test_hohmann_leo_to_geo():
    R_EARTH = 6371.0
    r1 = R_EARTH + 400.0
    r2 = R_EARTH + 35786.0
    result = hohmann_transfer(r1, r2)
    assert 3.8 < result.dv_total < 4.1
    assert 4.5 * 3600 < result.tof < 6.0 * 3600
    print(f"  ✓ Hohmann LEO->GEO: {result}")


def test_hohmann_symmetric():
    r1, r2 = 7000.0, 8000.0
    res_fwd = hohmann_transfer(r1, r2)
    res_bwd = hohmann_transfer(r2, r1)
    assert abs(res_fwd.dv_total - res_bwd.dv_total) < 1e-8
    print(f"  ✓ Hohmann symmetry: {res_fwd.dv_total:.6f} km/s both ways")


def test_hohmann_same_orbit():
    r = 7000.0
    result = hohmann_transfer(r, r)
    assert result.dv_total < 1e-10
    print(f"  ✓ Hohmann same orbit: dv = {result.dv_total:.2e} km/s")


def test_lambert_two_orbits():
    R_EARTH = 6371.0
    r1 = np.array([R_EARTH + 400.0, 0.0, 0.0])
    r2 = np.array([0.0, R_EARTH + 800.0, 0.0])
    result = lambert_transfer(r1, r2, tof=3600.0)
    assert result.transfer_type == "lambert"
    assert result.dv_total > 0
    print(f"  ✓ Lambert LEO->MEO: {result}")


if __name__ == "__main__":
    tests = [
        test_hohmann_leo_to_geo,
        test_hohmann_symmetric,
        test_hohmann_same_orbit,
        test_lambert_two_orbits,
    ]
    print("=== Tests: core/maneuver ===")
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")