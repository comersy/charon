"""Target satellite definition for Charon missions.

A Target represents a satellite that needs servicing — it wraps
an SGP4 propagator with mission-specific metadata: how much fuel
to deliver, priority, and deadline.

All units: km, km/s, seconds, kg.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.sgp4_propagator import SGP4Propagator, OrbitalState


@dataclass
class Target:
    """A satellite that needs on-orbit servicing.

    Attributes:
        name:          Human-readable identifier (e.g. 'STARLINK-1234').
        propagator:    SGP4Propagator instance built from the satellite's TLE.
        fuel_needed:   Propellant to be delivered (kg).
        priority:      Mission priority (1 = highest). Used by the optimizer.
        deadline:      Optional UTC datetime by which servicing must occur.
        notes:         Free-form notes (failure mode, operator contact, etc.).
    """
    name: str
    propagator: SGP4Propagator
    fuel_needed: float
    priority: int = 1
    deadline: Optional[datetime] = None
    notes: str = ""

    def __post_init__(self):
        if self.fuel_needed < 0:
            raise ValueError(f"fuel_needed must be >= 0, got {self.fuel_needed}")
        if self.priority < 1:
            raise ValueError(f"priority must be >= 1, got {self.priority}")

    # ------------------------------------------------------------------
    # Orbital state
    # ------------------------------------------------------------------

    def state_at(self, t: datetime) -> OrbitalState:
        """Return the orbital state of this target at time t.

        Args:
            t: UTC datetime (must be timezone-aware).

        Returns:
            OrbitalState with position (km) and velocity (km/s).
        """
        return self.propagator.propagate(t)

    def position_at(self, t: datetime) -> np.ndarray:
        """Return position vector [x, y, z] in km at time t."""
        return self.state_at(t).r

    def altitude_at(self, t: datetime) -> float:
        """Return altitude above Earth's surface in km at time t."""
        return self.state_at(t).altitude

    # ------------------------------------------------------------------
    # Deadline helpers
    # ------------------------------------------------------------------

    def has_deadline(self) -> bool:
        """Return True if a servicing deadline is set."""
        return self.deadline is not None

    def time_to_deadline(self, t: datetime) -> Optional[float]:
        """Seconds remaining until deadline from time t.

        Returns None if no deadline is set.
        Returns a negative value if the deadline has passed.
        """
        if self.deadline is None:
            return None
        return (self.deadline - t).total_seconds()

    def is_overdue(self, t: datetime) -> bool:
        """Return True if the deadline has passed at time t."""
        ttd = self.time_to_deadline(t)
        return ttd is not None and ttd < 0

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def summary(self, t: Optional[datetime] = None) -> str:
        """Human-readable summary, optionally with current orbital state."""
        lines = [
            f"Target '{self.name}'",
            f"  Priority     : {self.priority}",
            f"  Fuel needed  : {self.fuel_needed:.1f} kg",
            f"  Deadline     : {self.deadline.isoformat() if self.deadline else 'none'}",
        ]
        if t is not None:
            state = self.state_at(t)
            lines.append(f"  Altitude     : {state.altitude:.1f} km")
            lines.append(f"  Speed        : {state.speed:.3f} km/s")
            if self.has_deadline():
                ttd = self.time_to_deadline(t)
                lines.append(f"  Time to deadline: {ttd / 3600:.1f} h")
        if self.notes:
            lines.append(f"  Notes        : {self.notes}")
        return "\n".join(lines)

    def __repr__(self):
        return (
            f"Target(name='{self.name}', "
            f"fuel_needed={self.fuel_needed:.1f} kg, "
            f"priority={self.priority})"
        )


def target_from_tle(
    tle_block: str,
    fuel_needed: float,
    priority: int = 1,
    deadline: Optional[datetime] = None,
    notes: str = "",
) -> Target:
    """Convenience constructor: build a Target directly from a TLE string.

    Args:
        tle_block:   2- or 3-line TLE string.
        fuel_needed: Propellant to deliver (kg).
        priority:    Mission priority (1 = highest).
        deadline:    Optional UTC servicing deadline.
        notes:       Free-form notes.

    Returns:
        Target instance ready to use in a mission.

    Example:
        t = target_from_tle(tle_block=ISS_TLE, fuel_needed=150.0, priority=1)
    """
    prop = SGP4Propagator.from_tle_string(tle_block)
    name = prop.name or f"NORAD-{prop._sat.satnum}"
    return Target(
        name=name,
        propagator=prop,
        fuel_needed=fuel_needed,
        priority=priority,
        deadline=deadline,
        notes=notes,
    )