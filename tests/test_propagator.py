"""Unit tests for core/propagator."""
import sys
sys.path.insert(0, "/home/claude/charon")

from datetime import datetime, timezone, timedelta
import numpy as np

from core.propagator import SGP4Propagator, OrbitalState

# Real ISS TLE (epoch ~2024)
ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00001234  00000+0  12345-4 0  9990
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815764429440"""


def test_basic_propagation():
    prop = SGP4Propagator.from_tle_string(ISS_TLE)
    t = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    state = prop.propagate(t)

    assert isinstance(state, OrbitalState)
    assert 350 < state.altitude < 450, f"Unexpected altitude: {state.altitude:.1f} km"
    assert 7.0 < state.speed < 8.5, f"Unexpected speed: {state.speed:.3f} km/s"
    print(f"  ✓ Basic propagation: {state}")


def test_propagate_many():
    prop = SGP4Propagator.from_tle_string(ISS_TLE)
    t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    times = [t0 + timedelta(minutes=i * 10) for i in range(6)]
    states = prop.propagate_many(times)

    assert len(states) == 6
    assert not np.allclose(states[0].r, states[1].r), "Positions should differ over time"
    print(f"  ✓ propagate_many: {len(states)} states generated")


def test_naive_datetime_raises():
    prop = SGP4Propagator.from_tle_string(ISS_TLE)
    try:
        prop.propagate(datetime(2024, 1, 1, 12, 0, 0))
        assert False, "Should have raised ValueError"
    except ValueError:
        print("  ✓ Naive datetime correctly rejected")


def test_repr():
    prop = SGP4Propagator.from_tle_string(ISS_TLE)
    r = repr(prop)
    assert "SGP4Propagator" in r and "ISS" in r
    print(f"  ✓ repr: {r}")


if __name__ == "__main__":
    tests = [test_basic_propagation, test_propagate_many,
             test_naive_datetime_raises, test_repr]
    print("=== Tests: core/propagator ===")
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")