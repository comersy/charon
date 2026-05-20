"""Unit tests for mission/target.py."""
import sys
sys.path.insert(0, ".")

from datetime import datetime, timezone, timedelta
from mission.target import Target, target_from_tle
from core.sgp4_propagator import SGP4Propagator

ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00001234  00000+0  12345-4 0  9990
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815764429440"""

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_target_from_tle():
    t = target_from_tle(ISS_TLE, fuel_needed=150.0, priority=1)
    assert t.name == "ISS (ZARYA)"
    assert t.fuel_needed == 150.0
    assert t.priority == 1
    print(f"  ✓ target_from_tle: {t}")


def test_state_at():
    t = target_from_tle(ISS_TLE, fuel_needed=150.0)
    state = t.state_at(T0)
    assert 350 < state.altitude < 450
    print(f"  ✓ state_at: alt={state.altitude:.1f} km")


def test_deadline_logic():
    deadline = T0 + timedelta(hours=24)
    t = target_from_tle(ISS_TLE, fuel_needed=150.0, deadline=deadline)

    assert t.has_deadline()
    assert not t.is_overdue(T0)
    assert t.is_overdue(T0 + timedelta(hours=25))

    ttd = t.time_to_deadline(T0)
    assert abs(ttd - 86400.0) < 1.0
    print(f"  ✓ deadline logic: {ttd / 3600:.1f} h to deadline")


def test_no_deadline():
    t = target_from_tle(ISS_TLE, fuel_needed=150.0)
    assert not t.has_deadline()
    assert t.time_to_deadline(T0) is None
    print(f"  ✓ no deadline: OK")


def test_invalid_fuel():
    prop = SGP4Propagator.from_tle_string(ISS_TLE)
    try:
        Target(name="X", propagator=prop, fuel_needed=-10.0)
        assert False, "Should have raised ValueError"
    except ValueError:
        print(f"  ✓ negative fuel_needed rejected")


def test_summary():
    t = target_from_tle(ISS_TLE, fuel_needed=150.0, notes="Battery degraded")
    s = t.summary(T0)
    assert "ISS" in s and "Battery degraded" in s
    print(f"  ✓ summary OK")


if __name__ == "__main__":
    tests = [
        test_target_from_tle,
        test_state_at,
        test_deadline_logic,
        test_no_deadline,
        test_invalid_fuel,
        test_summary,
    ]
    print("=== Tests: mission/target ===")
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")