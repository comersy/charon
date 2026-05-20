"""Mission sequence definition for Charon.

A Sequence is an ordered list of Targets representing the visit order
of the servicer spacecraft. It computes the total delta-v cost of the
mission using Hohmann transfers as a fast approximation.

All units: km, km/s, seconds, kg.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from core.maneuver import hohmann_transfer, ManeuverResult
from core.spacecraft import Spacecraft
from mission.target import Target


@dataclass
class VisitRecord:
    """Result of a single visit in the sequence.

    Attributes:
        target:    The Target that was visited.
        maneuver:  ManeuverResult for the transfer to this target.
        t_arrival: Estimated UTC arrival time.
        fuel_used: Propellant consumed for this transfer (kg).
    """
    target: Target
    maneuver: ManeuverResult
    t_arrival: datetime
    fuel_used: float

    def __repr__(self):
        return (
            f"VisitRecord(target='{self.target.name}', "
            f"dv={self.maneuver.dv_total:.4f} km/s, "
            f"fuel={self.fuel_used:.2f} kg, "
            f"arrival={self.t_arrival.isoformat()})"
        )


class Sequence:
    """An ordered list of targets defining a servicing mission.

    Evaluates the total delta-v and fuel cost of visiting targets
    in a given order, starting from a depot orbit.

    Example:
        seq = Sequence(targets=[t1, t2, t3], spacecraft=sc, t_start=T0)
        print(seq.total_dv())
        print(seq.is_feasible())
    """

    def __init__(
        self,
        targets: list[Target],
        spacecraft: Spacecraft,
        t_start: datetime,
        depot_altitude: float = 400.0,
    ):
        """
        Args:
            targets:          Ordered list of Target objects to visit.
            spacecraft:       Servicer spacecraft (used for fuel checks).
            t_start:          UTC start time of the mission.
            depot_altitude:   Altitude of the servicer's initial parking
                              orbit in km. Default: 400 km (ISS-like LEO).
        """
        self.targets = targets
        self.spacecraft = spacecraft
        self.t_start = t_start
        self.depot_altitude = depot_altitude
        self._records: Optional[list[VisitRecord]] = None

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> list[VisitRecord]:
        """Compute the full sequence: transfers, arrivals, fuel used.

        Uses Hohmann transfers between consecutive target altitudes as a
        fast approximation. Lambert solvers are used in the optimizer for
        higher fidelity.

        Returns:
            List of VisitRecord, one per target.
        """
        records = []
        R_EARTH = 6371.0
        t_current = self.t_start
        r_current = R_EARTH + self.depot_altitude  # km from Earth center

        for target in self.targets:
            state = target.state_at(t_current)
            r_target = np.linalg.norm(state.r)

            maneuver = hohmann_transfer(r_current, r_target)
            fuel_used = self.spacecraft.dv_to_fuel(maneuver.dv_total)

            t_arrival = t_current + timedelta(seconds=maneuver.tof)

            records.append(VisitRecord(
                target=target,
                maneuver=maneuver,
                t_arrival=t_arrival,
                fuel_used=fuel_used,
            ))

            # Update state for next leg
            r_current = r_target
            t_current = t_arrival

        self._records = records
        return records

    # ------------------------------------------------------------------
    # Cost metrics
    # ------------------------------------------------------------------

    def total_dv(self) -> float:
        """Total delta-v cost of the sequence (km/s)."""
        records = self._records or self.evaluate()
        return sum(r.maneuver.dv_total for r in records)

    def total_fuel(self) -> float:
        """Total propellant consumed across all transfers (kg)."""
        records = self._records or self.evaluate()
        return sum(r.fuel_used for r in records)

    def total_duration(self) -> float:
        """Total mission duration in seconds."""
        records = self._records or self.evaluate()
        return sum(r.maneuver.tof for r in records)

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def is_feasible(self) -> bool:
        """Return True if the spacecraft has enough fuel for the full sequence
        and no target deadline is missed."""
        records = self._records or self.evaluate()

        cumulative_fuel = 0.0
        for r in records:
            cumulative_fuel += r.fuel_used
            if cumulative_fuel > self.spacecraft.fuel_remaining:
                return False
            if r.target.is_overdue(r.t_arrival):
                return False

        return True

    def infeasibility_reasons(self) -> list[str]:
        """Return a list of reasons why the sequence is infeasible (if any)."""
        records = self._records or self.evaluate()
        reasons = []
        cumulative_fuel = 0.0

        for r in records:
            cumulative_fuel += r.fuel_used
            if cumulative_fuel > self.spacecraft.fuel_remaining:
                reasons.append(
                    f"Insufficient fuel after visiting '{r.target.name}': "
                    f"need {cumulative_fuel:.1f} kg, "
                    f"have {self.spacecraft.fuel_remaining:.1f} kg"
                )
            if r.target.is_overdue(r.t_arrival):
                reasons.append(
                    f"Deadline missed for '{r.target.name}': "
                    f"arrived {r.t_arrival.isoformat()}, "
                    f"deadline {r.target.deadline.isoformat()}"
                )

        return reasons

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable mission summary."""
        records = self._records or self.evaluate()
        lines = [
            f"Mission Sequence — {len(self.targets)} targets",
            f"  Start time   : {self.t_start.isoformat()}",
            f"  Depot orbit  : {self.depot_altitude:.0f} km",
            f"  Total dv     : {self.total_dv():.4f} km/s",
            f"  Total fuel   : {self.total_fuel():.2f} kg",
            f"  Duration     : {self.total_duration() / 3600:.2f} h",
            f"  Feasible     : {self.is_feasible()}",
            "",
            "  Visit order:",
        ]
        for i, r in enumerate(records, 1):
            lines.append(
                f"    {i}. {r.target.name:20s} "
                f"dv={r.maneuver.dv_total:.3f} km/s  "
                f"fuel={r.fuel_used:.1f} kg  "
                f"arrival={r.t_arrival.strftime('%Y-%m-%dT%H:%M')}"
            )
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"Sequence(targets={[t.name for t in self.targets]}, "
            f"dv={self.total_dv():.4f} km/s)"
        )