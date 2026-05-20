"""Unit tests for core/spacecraft.py."""
import sys
sys.path.insert(0, ".")

from datetime import datetime, timezone
import numpy as np

from core.spacecraft import Spacecraft
from core.maneuver import hohmann_transfer


T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_basic_state():
    sc = Spacecraft(dry_mass=500.0, fuel_mass=2000.0, isp=310.0)
    assert sc.total_mass == 2500.0
    assert sc.fuel_remaining == 2000.0
    assert sc.fuel_fraction == 1.0
    print(f"  ✓ Basic state: {sc}")


def test_tsiolkovsky():
    """Fuel consumption should match rocket equation manually."""
    sc = Spacecraft(dry_mass=500.0, fuel_mass=2000.0, isp=310.0)
    dv = 1.0  # km/s
    g0 = 9.80665e-3
    expected = sc.total_mass * (1 - np.exp(-dv / (310.0 * g0)))
    assert abs(sc.dv_to_fuel(dv) - expected) < 1e-10
    print(f"  ✓ Tsiolkovsky: {expected:.2f} kg for dv=1.0 km/s")


def test_apply_maneuver():
    sc = Spacecraft(dry_mass=500.0, fuel_mass=2000.0, isp=310.0)
    maneuver = hohmann_transfer(6771.0, 7171.0)  # 400->800 km
    record = sc.apply_maneuver(maneuver, T0, target="SAT-01")

    assert sc.fuel_remaining < 2000.0
    assert record.target == "SAT-01"
    assert len(sc.history) == 1
    print(f"  ✓ apply_maneuver: used {record.fuel_used:.2f} kg, target={record.target}")


def test_insufficient_fuel():
    sc = Spacecraft(dry_mass=500.0, fuel_mass=1.0, isp=310.0)
    maneuver = hohmann_transfer(6771.0, 42164.0)  # LEO->GEO, need ~3.9 km/s
    try:
        sc.apply_maneuver(maneuver, T0)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"  ✓ Insufficient fuel caught: {e}")


def test_max_dv():
    sc = Spacecraft(dry_mass=500.0, fuel_mass=2000.0, isp=310.0)
    dv_max = sc.max_dv()
    assert dv_max > 0
    # After spending all fuel, max_dv should be 0
    sc._fuel_mass = 0.0
    assert sc.max_dv() == 0.0
    print(f"  ✓ max_dv: {dv_max:.4f} km/s at full fuel")


def test_status():
    sc = Spacecraft(dry_mass=500.0, fuel_mass=2000.0, isp=310.0, name="Charon-1")
    s = sc.status()
    assert "Charon-1" in s
    print(f"  ✓ status output OK")


if __name__ == "__main__":
    tests = [
        test_basic_state,
        test_tsiolkovsky,
        test_apply_maneuver,
        test_insufficient_fuel,
        test_max_dv,
        test_status,
    ]
    print("=== Tests: core/spacecraft ===")
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")