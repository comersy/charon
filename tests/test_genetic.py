"""Unit tests for optimizer/genetic.py."""
import sys
sys.path.insert(0, ".")

from datetime import datetime, timezone
from core.spacecraft import Spacecraft
from mission.target import target_from_tle
from optimizer.genetic import GeneticOptimizer, compute_fitness, order_crossover, Individual

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


def make_targets():
    return [
        target_from_tle(ISS_TLE,        fuel_needed=120.0, priority=1),
        target_from_tle(STARLINK_TLE,   fuel_needed=80.0,  priority=2),
        target_from_tle(STARLINK_TLE_2, fuel_needed=60.0,  priority=3),
    ]


def test_compute_fitness():
    targets = make_targets()
    sc = Spacecraft(dry_mass=500.0, fuel_mass=3000.0, isp=310.0)
    fitness = compute_fitness([0, 1, 2], targets, sc, T0)
    assert fitness > 0
    print(f"  ✓ compute_fitness: {fitness:.4f} km/s")


def test_order_crossover():
    a = Individual(genes=[0, 1, 2, 3, 4])
    b = Individual(genes=[4, 3, 2, 1, 0])
    child = order_crossover(a, b)
    assert sorted(child.genes) == [0, 1, 2, 3, 4]
    assert len(child.genes) == 5
    print(f"  ✓ order_crossover: {child.genes}")


def test_optimizer_runs():
    targets = make_targets()
    sc = Spacecraft(dry_mass=500.0, fuel_mass=3000.0, isp=310.0)
    optimizer = GeneticOptimizer(
        targets=targets,
        spacecraft=sc,
        t_start=T0,
        pop_size=20,
        n_generations=10,
        seed=42,
    )
    result = optimizer.run()
    assert len(result.best_sequence) == 3
    assert result.best_dv > 0
    assert len(result.history) == 10
    print(f"  ✓ optimizer ran: best_dv={result.best_dv:.4f} km/s, feasible={result.feasible}")


def test_optimizer_improves():
    """Best fitness should not increase over generations."""
    targets = make_targets()
    sc = Spacecraft(dry_mass=500.0, fuel_mass=3000.0, isp=310.0)
    optimizer = GeneticOptimizer(
        targets=targets,
        spacecraft=sc,
        t_start=T0,
        pop_size=30,
        n_generations=20,
        seed=42,
    )
    result = optimizer.run()
    assert result.history[-1] <= result.history[0]
    print(f"  ✓ optimizer improves: {result.history[0]:.4f} -> {result.history[-1]:.4f} km/s")


def test_callback():
    targets = make_targets()
    sc = Spacecraft(dry_mass=500.0, fuel_mass=3000.0, isp=310.0)
    log = []
    optimizer = GeneticOptimizer(
        targets=targets, spacecraft=sc, t_start=T0,
        pop_size=10, n_generations=5, seed=0,
        on_generation=lambda g, f: log.append((g, f)),
    )
    optimizer.run()
    assert len(log) == 5
    print(f"  ✓ callback fired {len(log)} times")


if __name__ == "__main__":
    tests = [
        test_compute_fitness,
        test_order_crossover,
        test_optimizer_runs,
        test_optimizer_improves,
        test_callback,
    ]
    print("=== Tests: optimizer/genetic ===")
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"  ✗ {test.__name__}: {e}")
            failed += 1
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed.")