"""Unit tests for simulation/timeline.py."""
import sys
sys.path.insert(0, "/home/claude/charon")

from datetime import datetime, timezone
from core.spacecraft import Spacecraft
from mission.target import target_from_tle
from simulation.timeline import MissionPlanner, MissionTimeline

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

ISS_TLE = """ISS (ZARYA)
1 25544U 98067A   24001.50000000  .00001234  00000+0  12345-4 0  9990
2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815764429440"""

STARLINK_TLE = """STARLINK-1234
1 45012U 20001A   24001.50000000  .00001000  00000+0  10000-4 0  9991
2 45012  53.0000 100.0000 0001000  90.0000 270.0000 15.06000000000001"""

STARLINK_TLE_2 = """STARLINK-5678
1 45013U 20001B   24001.50000000  .00001100  00000+0  11000-4 0  9992
2 45013  53.0000 120.0000 0001200  80.0000 280.0000 15.07000000000001"""


def make_inputs():
    targets = [
        target_from_tle(ISS_TLE,        fuel_needed=120.0, priority=1),
        target_from_tle(STARLINK_TLE,   fuel_needed=80.0,  priority=2),
        target_from_tle(STARLINK_TLE_2, fuel_needed=60.0,  priority=3),
    ]
    sc = Spacecraft(dry_mass=500.0, fuel_mass=3000.0, isp=310.0, name="Charon-1")
    return targets, sc


def test_planner_runs():
    targets, sc = make_inputs()
    planner = MissionPlanner(
        targets=targets, spacecraft=sc, t_start=T0,
        pop_size=10, n_generations=5, seed=42,
    )
    timeline = planner.run()
    assert isinstance(timeline, MissionTimeline)
    assert len(timeline.events) > 0
    assert timeline.total_dv > 0
    print(f"  ✓ planner runs: {timeline}")


def test_timeline_has_all_event_kinds():
    targets, sc = make_inputs()
    planner = MissionPlanner(
        targets=targets, spacecraft=sc, t_start=T0,
        pop_size=10, n_generations=5, seed=42,
    )
    timeline = planner.run()
    kinds = {e.kind for e in timeline.events}
    assert "depart"    in kinds
    assert "transfer"  in kinds
    assert "rendezvous" in kinds
    assert "dock"      in kinds
    print(f"  ✓ event kinds: {kinds}")


def test_rendezvous_results_populated():
    targets, sc = make_inputs()
    planner = MissionPlanner(
        targets=targets, spacecraft=sc, t_start=T0,
        pop_size=10, n_generations=5, seed=42,
    )
    timeline = planner.run()
    assert len(timeline.rendezvous) == len(targets)
    print(f"  ✓ rendezvous results: {len(timeline.rendezvous)} entries")


def test_summary_output():
    targets, sc = make_inputs()
    planner = MissionPlanner(
        targets=targets, spacecraft=sc, t_start=T0,
        pop_size=10, n_generations=5, seed=42,
    )
    timeline = planner.run()
    s = timeline.summary()
    assert "CHARON" in s
    assert "Total Δv" in s
    print(f"  ✓ summary OK ({len(s)} chars)")


def test_duration_positive():
    targets, sc = make_inputs()
    planner = MissionPlanner(
        targets=targets, spacecraft=sc, t_start=T0,
        pop_size=10, n_generations=5, seed=42,
    )
    timeline = planner.run()
    assert timeline.duration_hours > 0
    print(f"  ✓ duration: {timeline.duration_hours:.2f} h")


if __name__ == "__main__":
    tests = [
        test_planner_runs,
        test_timeline_has_all_event_kinds,
        test_rendezvous_results_populated,
        test_summary_output,
        test_duration_positive,
    ]
    print("=== Tests: simulation/timeline ===")
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")