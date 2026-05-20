"""End-to-end mission timeline for Charon.

Chains the full pipeline:
    1. Genetic optimizer finds the best visit order.
    2. Sequence evaluates transfers between targets.
    3. RendezvousSimulator models the final approach for each target.

All units: km, km/s, seconds, kg.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from core.spacecraft import Spacecraft
from mission.target import Target
from mission.sequence import Sequence, VisitRecord
from optimizer.genetic import GeneticOptimizer, OptimizationResult
from simulation.rendezvous import RendezvousSimulator, RendezvousResult


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class MissionEvent:
    """A single timestamped event in the mission timeline.

    Attributes:
        t:           UTC time of the event.
        kind:        Event type: 'depart', 'transfer', 'rendezvous', 'dock'.
        target_name: Name of the relevant target (if applicable).
        description: Human-readable description.
        dv:          Delta-v cost of this event (km/s). 0 if not a burn.
        fuel_used:   Propellant consumed (kg). 0 if not a burn.
    """
    t: datetime
    kind: str
    target_name: str = ""
    description: str = ""
    dv: float = 0.0
    fuel_used: float = 0.0

    def __repr__(self):
        return (
            f"[{self.t.strftime('%Y-%m-%dT%H:%M')}] "
            f"{self.kind.upper():12s} {self.target_name:20s} "
            f"dv={self.dv:.4f} km/s  fuel={self.fuel_used:.1f} kg"
        )


@dataclass
class MissionTimeline:
    """Complete end-to-end mission timeline.

    Attributes:
        events:          Ordered list of MissionEvents.
        optimization:    Result from the genetic optimizer.
        visit_records:   Transfer details for each target visit.
        rendezvous:      Rendezvous simulation result per target.
        spacecraft:      Servicer spacecraft (final state after mission).
        t_start:         Mission start time.
        t_end:           Mission end time.
    """
    events: list[MissionEvent]
    optimization: OptimizationResult
    visit_records: list[VisitRecord]
    rendezvous: dict[str, RendezvousResult]
    spacecraft: Spacecraft
    t_start: datetime
    t_end: datetime

    @property
    def total_dv(self) -> float:
        """Total delta-v across all events (km/s)."""
        return sum(e.dv for e in self.events)

    @property
    def total_fuel(self) -> float:
        """Total propellant consumed (kg)."""
        return sum(e.fuel_used for e in self.events)

    @property
    def duration_hours(self) -> float:
        """Total mission duration in hours."""
        return (self.t_end - self.t_start).total_seconds() / 3600

    @property
    def n_docked(self) -> int:
        """Number of successful dockings."""
        return sum(1 for r in self.rendezvous.values() if r.success)

    def summary(self) -> str:
        """Human-readable mission summary."""
        lines = [
            "=" * 60,
            "  CHARON — Mission Timeline Summary",
            "=" * 60,
            f"  Start          : {self.t_start.isoformat()}",
            f"  End            : {self.t_end.isoformat()}",
            f"  Duration       : {self.duration_hours:.1f} h",
            f"  Targets        : {len(self.visit_records)}",
            f"  Docked         : {self.n_docked}/{len(self.visit_records)}",
            f"  Total Δv       : {self.total_dv:.4f} km/s",
            f"  Total fuel     : {self.total_fuel:.1f} kg",
            f"  Fuel remaining : {self.spacecraft.fuel_remaining:.1f} kg",
            f"  Optimizer gens : {self.optimization.generations}",
            f"  Feasible       : {self.optimization.feasible}",
            "",
            "  Event log:",
        ]
        for e in self.events:
            lines.append(f"    {e}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"MissionTimeline("
            f"targets={len(self.visit_records)}, "
            f"dv={self.total_dv:.4f} km/s, "
            f"duration={self.duration_hours:.1f} h)"
        )


# ------------------------------------------------------------------
# Timeline builder
# ------------------------------------------------------------------

class MissionPlanner:
    """Builds a complete mission timeline from targets and a spacecraft.

    Runs the genetic optimizer to find the best visit order, then
    simulates each transfer and rendezvous leg end-to-end.

    Example:
        planner = MissionPlanner(
            targets=targets,
            spacecraft=spacecraft,
            t_start=T0,
            pop_size=50,
            n_generations=100,
        )
        timeline = planner.run()
        print(timeline.summary())
    """

    def __init__(
        self,
        targets: list[Target],
        spacecraft: Spacecraft,
        t_start: datetime,
        depot_altitude: float = 400.0,
        pop_size: int = 50,
        n_generations: int = 100,
        rendezvous_range: float = 5.0,
        rendezvous_threshold: float = 0.01,
        seed: Optional[int] = None,
    ):
        """
        Args:
            targets:              Targets to service.
            spacecraft:           Servicer spacecraft.
            t_start:              Mission start time (UTC).
            depot_altitude:       Initial parking orbit altitude (km).
            pop_size:             Genetic algorithm population size.
            n_generations:        Number of GA generations.
            rendezvous_range:     Initial range for rendezvous sim (km).
            rendezvous_threshold: Docking threshold (km).
            seed:                 Random seed for reproducibility.
        """
        self.targets = targets
        self.spacecraft = spacecraft
        self.t_start = t_start
        self.depot_altitude = depot_altitude
        self.pop_size = pop_size
        self.n_generations = n_generations
        self.rendezvous_range = rendezvous_range
        self.rendezvous_threshold = rendezvous_threshold
        self.seed = seed

    def run(self) -> MissionTimeline:
        """Run the full mission planning pipeline.

        Returns:
            MissionTimeline with all events, transfers, and rendezvous results.
        """
        # Step 1 — Optimize visit order
        optimizer = GeneticOptimizer(
            targets=self.targets,
            spacecraft=self.spacecraft,
            t_start=self.t_start,
            depot_altitude=self.depot_altitude,
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            seed=self.seed,
        )
        opt_result = optimizer.run()

        # Step 2 — Evaluate sequence with optimized order
        sequence = Sequence(
            targets=opt_result.best_sequence,
            spacecraft=self.spacecraft,
            t_start=self.t_start,
            depot_altitude=self.depot_altitude,
        )
        visit_records = sequence.evaluate()

        # Step 3 — Build timeline events + simulate each rendezvous
        events: list[MissionEvent] = []
        rendezvous_results: dict[str, RendezvousResult] = {}

        events.append(MissionEvent(
            t=self.t_start,
            kind="depart",
            description=f"Servicer departs depot at {self.depot_altitude:.0f} km",
        ))

        for record in visit_records:
            target = record.target

            # Transfer burn
            events.append(MissionEvent(
                t=record.t_arrival - timedelta(seconds=record.maneuver.tof),
                kind="transfer",
                target_name=target.name,
                description=f"Hohmann transfer to {target.name}",
                dv=record.maneuver.dv_total,
                fuel_used=record.fuel_used,
            ))

            # Rendezvous simulation
            state = target.state_at(record.t_arrival)
            sim = RendezvousSimulator(
                initial_range=self.rendezvous_range,
                target_altitude=state.altitude,
                approach_velocity=0.005,
                docking_threshold=self.rendezvous_threshold,
                t_start=record.t_arrival,
            )
            rdv_result = sim.run()
            rendezvous_results[target.name] = rdv_result

            events.append(MissionEvent(
                t=record.t_arrival,
                kind="rendezvous",
                target_name=target.name,
                description=f"Final approach to {target.name}",
                dv=rdv_result.total_dv,
                fuel_used=self.spacecraft.dv_to_fuel(rdv_result.total_dv),
            ))

            # Docking
            t_dock = record.t_arrival + timedelta(
                seconds=rdv_result.states[-1].t.timestamp() - record.t_arrival.timestamp()
                if rdv_result.states else 0
            )
            events.append(MissionEvent(
                t=record.t_arrival,
                kind="dock",
                target_name=target.name,
                description=f"{'Docked' if rdv_result.success else 'FAILED'} at {target.name} "
                            f"— delivering {target.fuel_needed:.0f} kg",
            ))

            # Apply transfer burn to spacecraft fuel budget
            if self.spacecraft.can_perform(record.maneuver.dv_total):
                self.spacecraft.apply_maneuver(record.maneuver, record.t_arrival, target.name)

        t_end = visit_records[-1].t_arrival if visit_records else self.t_start

        return MissionTimeline(
            events=sorted(events, key=lambda e: e.t),
            optimization=opt_result,
            visit_records=visit_records,
            rendezvous=rendezvous_results,
            spacecraft=self.spacecraft,
            t_start=self.t_start,
            t_end=t_end,
        )