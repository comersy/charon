"""Unit tests for mission/sequence.py."""
import sys
sys.path.insert(0, ".")

from datetime import datetime, timezone, timedelta
from core.spacecraft import Spacecraft
from mission.target import target_from_tle
from mission.sequence import Sequence

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00001234  00000+0  12345-4 0  9990
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815764429440"""

STARLINK_TLE = """STARLINK-1234
1 45012U 20001A   24001.50000000  .00001000  00000+0  10000-4 0  9991
2 45012  53.0000 100.0000 0001000  90.0000 270.0000 15.06000000000001"""


def test_basic_sequence():
    sc = Spacecraft(dry_mass=500.0, fuel_mass=3000.0, isp=310.0)
    t1 = target_from_tle(ISS_TLE, fuel_needed=100.0, priority=1)
    t2 = target_from_tle(STARLINK_TLE, fuel_needed=80.0, priority=2)

    seq = Sequence(targets=[t1, t2], spacecraft=sc, t_start=T0)
    records = seq.evaluate()

    assert len(records) == 2
    assert seq.total_dv() > 0
    assert seq.total_fuel() > 0
    assert seq.total_duration() > 0
    print(f"  ✓ Basic sequence: dv={seq.total_dv():.4f} km/s")


def test_feasible_sequence():
    sc = Spacecraft(dry_mass=500.0, fuel_mass=3000.0, isp=310.0)
    t1 = target_from_tle(ISS_TLE, fuel_needed=100.0)
    seq = Sequence(targets=[t1], spacecraft=sc, t_start=T0)

    assert seq.is_feasible()
    print(f"  ✓ Feasible sequence: {seq.is_feasible()}")


def test_infeasible_no_fuel():
    sc = Spacecraft(dry_mass=500.0, fuel_mass=0.1, isp=310.0)
    t1 = target_from_tle(ISS_TLE, fuel_needed=100.0)
    seq = Sequence(targets=[t1], spacecraft=sc, t_start=T0)

    assert not seq.is_feasible()
    reasons = seq.infeasibility_reasons()
    assert len(reasons) > 0
    print(f"  ✓ Infeasible (no fuel): {reasons[0]}")


def test_infeasible_deadline():
    sc = Spacecraft(dry_mass=500.0, fuel_mass=3000.0, isp=310.0)
    deadline = T0 + timedelta(seconds=1)  # impossible deadline
    t1 = target_from_tle(ISS_TLE, fuel_needed=100.0, deadline=deadline)
    seq = Sequence(targets=[t1], spacecraft=sc, t_start=T0)

    assert not seq.is_feasible()
    reasons = seq.infeasibility_reasons()
    assert any("Deadline" in r for r in reasons)
    print(f"  ✓ Infeasible (deadline): {reasons[0][:60]}...")


def test_summary():
    sc = Spacecraft(dry_mass=500.0, fuel_mass=3000.0, isp=310.0)
    t1 = target_from_tle(ISS_TLE, fuel_needed=100.0)
    seq = Sequence(targets=[t1], spacecraft=sc, t_start=T0)
    s = seq.summary()
    assert "ISS" in s and "Total dv" in s
    print(f"  ✓ summary OK")


if __name__ == "__main__":
    tests = [
        test_basic_sequence,
        test_feasible_sequence,
        test_infeasible_no_fuel,
        test_infeasible_deadline,
        test_summary,
    ]
    print("=== Tests: mission/sequence ===")
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")