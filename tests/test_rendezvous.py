"""Unit tests for simulation/rendezvous.py."""
import sys
sys.path.insert(0, ".")

import numpy as np
from datetime import datetime, timezone

from simulation.rendezvous import (
    cw_propagate, mean_motion, RendezvousSimulator, RelativeState
)

T0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_mean_motion():
    """ISS at ~410 km should have n ~ 0.00113 rad/s."""
    n = mean_motion(410.0)
    assert 0.001 < n < 0.0015
    print(f"  ✓ mean_motion(410 km) = {n:.6f} rad/s")


def test_cw_propagate_stationary():
    """A target at origin with zero velocity stays near origin."""
    r0 = np.array([0.0, 0.0, 0.0])
    v0 = np.array([0.0, 0.0, 0.0])
    n = mean_motion(410.0)
    r1, v1 = cw_propagate(r0, v0, dt=60.0, n=n)
    assert np.linalg.norm(r1) < 1e-10
    print(f"  ✓ cw_propagate stationary: range={np.linalg.norm(r1):.2e} km")


def test_cw_propagate_motion():
    """An along-track offset should produce drift over time."""
    r0 = np.array([-1.0, 0.0, 0.0])
    v0 = np.array([0.01, 0.0, 0.0])
    n = mean_motion(410.0)
    r1, v1 = cw_propagate(r0, v0, dt=300.0, n=n)
    assert not np.allclose(r0, r1)
    print(f"  ✓ cw_propagate motion: {r0} -> {r1}")


def test_relative_state_range():
    state = RelativeState(
        t=T0,
        r=np.array([3.0, 4.0, 0.0]),
        v=np.array([-0.01, 0.0, 0.0]),
    )
    assert abs(state.range - 5.0) < 1e-10
    print(f"  ✓ RelativeState.range = {state.range:.1f} km")


def test_rendezvous_success():
    """Simulator should dock within max_duration."""
    sim = RendezvousSimulator(
        initial_range=2.0,
        target_altitude=410.0,
        approach_velocity=0.01,
        docking_threshold=0.05,
        max_duration=7200.0,
        dt=10.0,
        correction_interval=120.0,
        t_start=T0,
    )
    result = sim.run()
    assert result.success, f"Expected docking, final range={result.final_range:.4f} km"
    assert result.total_dv >= 0
    assert len(result.states) > 1
    print(f"  ✓ Rendezvous success: {result}")


def test_rendezvous_trajectory_length():
    """Trajectory should have at least a few states."""
    sim = RendezvousSimulator(initial_range=1.0, t_start=T0)
    result = sim.run()
    assert len(result.states) > 10
    print(f"  ✓ Trajectory length: {len(result.states)} states")


if __name__ == "__main__":
    tests = [
        test_mean_motion,
        test_cw_propagate_stationary,
        test_cw_propagate_motion,
        test_relative_state_range,
        test_rendezvous_success,
        test_rendezvous_trajectory_length,
    ]
    print("=== Tests: simulation/rendezvous ===")
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")