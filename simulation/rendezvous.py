"""Rendezvous simulation using Clohessy-Wiltshire (Hill) equations.

Models the relative motion between the servicer and a target satellite
in the Local Vertical Local Horizontal (LVLH) frame during the final
approach phase.

Reference:
    Clohessy, W.H. & Wiltshire, R.S. (1960). Terminal Guidance System
    for Satellite Rendezvous. Journal of the Aerospace Sciences, 27(9).

All units: km, km/s, seconds.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

from core.sgp4_propagator import OrbitalState


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class RelativeState:
    """Relative position and velocity in the LVLH frame.

    Axes (Hill frame):
        x: Along-track (positive in direction of motion)
        y: Cross-track (positive out of orbital plane)
        z: Radial (positive away from Earth)

    Attributes:
        t:   UTC timestamp.
        r:   Relative position [x, y, z] in km.
        v:   Relative velocity [vx, vy, vz] in km/s.
    """
    t: datetime
    r: np.ndarray
    v: np.ndarray

    @property
    def range(self) -> float:
        """Distance to target in km."""
        return float(np.linalg.norm(self.r))

    @property
    def range_rate(self) -> float:
        """Closing speed in km/s (negative = approaching)."""
        return float(np.dot(self.r, self.v) / self.range) if self.range > 0 else 0.0

    def __repr__(self):
        return (
            f"RelativeState(t={self.t.isoformat()}, "
            f"range={self.range:.3f} km, "
            f"range_rate={self.range_rate:.4f} km/s)"
        )


@dataclass
class RendezvousResult:
    """Full rendezvous trajectory from initial approach to docking.

    Attributes:
        states:         Time history of relative states.
        dv_corrections: List of (t, dv_vector) correction burns applied.
        success:        True if final range <= docking_threshold.
        final_range:    Range at end of simulation (km).
        total_dv:       Total delta-v from all correction burns (km/s).
    """
    states: list[RelativeState]
    dv_corrections: list[tuple[datetime, np.ndarray]]
    success: bool
    final_range: float
    total_dv: float

    def __repr__(self):
        return (
            f"RendezvousResult(success={self.success}, "
            f"final_range={self.final_range:.4f} km, "
            f"total_dv={self.total_dv:.4f} km/s, "
            f"steps={len(self.states)})"
        )


# ------------------------------------------------------------------
# CW propagator
# ------------------------------------------------------------------

def cw_propagate(
    r0: np.ndarray,
    v0: np.ndarray,
    dt: float,
    n: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate relative state using the Clohessy-Wiltshire equations.

    Valid for small relative distances (< ~50 km) on near-circular orbits.

    Args:
        r0: Initial relative position [x, y, z] in km.
        v0: Initial relative velocity [vx, vy, vz] in km/s.
        dt: Propagation time in seconds.
        n:  Mean motion of the target orbit (rad/s).

    Returns:
        (r1, v1): Propagated relative position and velocity.

    Reference:
        Schaub & Junkins (2018). Analytical Mechanics of Space Systems.
        Chapter 14, Eq. 14.68.
    """
    s = np.sin(n * dt)
    c = np.cos(n * dt)

    x0, y0, z0     = r0
    xd0, yd0, zd0  = v0

    # CW state transition — position
    x1 = (4 - 3*c) * x0 + s * xd0/n + 2*(1-c) * yd0/n
    y1 = 6*(s - n*dt) * x0 + y0 - 2*(1-c) * xd0/n + (4*s - 3*n*dt) * yd0/n
    z1 = z0 * c + zd0 * s / n

    # CW state transition — velocity
    xd1 = 3*n*s * x0 + c * xd0 + 2*s * yd0
    yd1 = 6*n*(c-1) * x0 - 2*s * xd0 + (4*c - 3) * yd0
    zd1 = -z0 * n * s + zd0 * c

    return np.array([x1, y1, z1]), np.array([xd1, yd1, zd1])


def mean_motion(altitude_km: float) -> float:
    """Compute orbital mean motion n (rad/s) for a circular orbit.

    Args:
        altitude_km: Orbit altitude above Earth surface in km.

    Returns:
        Mean motion in rad/s.
    """
    MU = 398600.4418
    R_EARTH = 6371.0
    r = R_EARTH + altitude_km
    return np.sqrt(MU / r**3)


# ------------------------------------------------------------------
# Rendezvous simulator
# ------------------------------------------------------------------

class RendezvousSimulator:
    """Simulate the final approach of a servicer to a target satellite.

    Uses CW equations for relative motion propagation. Applies
    periodic velocity corrections to drive the servicer toward
    the target (simple proportional guidance law).

    Example:
        sim = RendezvousSimulator(
            initial_range=5.0,       # km
            target_altitude=410.0,   # km
            approach_velocity=0.01,  # km/s
        )
        result = sim.run()
        print(result)
    """

    def __init__(
        self,
        initial_range: float = 5.0,
        target_altitude: float = 410.0,
        approach_velocity: float = 0.005,
        docking_threshold: float = 0.01,
        max_duration: float = 7200.0,
        dt: float = 10.0,
        correction_interval: float = 300.0,
        t_start: Optional[datetime] = None,
    ):
        """
        Args:
            initial_range:        Starting distance from target (km).
            target_altitude:      Target orbit altitude (km). Used to
                                  compute mean motion.
            approach_velocity:    Initial closing speed (km/s).
            docking_threshold:    Range at which docking is declared (km).
            max_duration:         Maximum simulation time (seconds).
            dt:                   Integration timestep (seconds).
            correction_interval:  Time between guidance burns (seconds).
            t_start:              UTC start time. Defaults to now.
        """
        self.initial_range = initial_range
        self.n = mean_motion(target_altitude)
        self.approach_velocity = approach_velocity
        self.docking_threshold = docking_threshold
        self.max_duration = max_duration
        self.dt = dt
        self.correction_interval = correction_interval
        self.t_start = t_start or datetime.now()

    def run(self) -> RendezvousResult:
        """Run the rendezvous simulation.

        Returns:
            RendezvousResult with full trajectory and docking outcome.
        """
        # Initial state: servicer is behind target along track
        r = np.array([-self.initial_range, 0.0, 0.0])
        v = np.array([self.approach_velocity, 0.0, 0.0])

        t = self.t_start
        states = [RelativeState(t=t, r=r.copy(), v=v.copy())]
        corrections = []
        total_dv = 0.0
        time_elapsed = 0.0
        time_since_correction = 0.0

        while time_elapsed < self.max_duration:
            r, v = cw_propagate(r, v, self.dt, self.n)
            time_elapsed += self.dt
            time_since_correction += self.dt
            t = self.t_start + timedelta(seconds=time_elapsed)

            # Proportional guidance correction
            if time_since_correction >= self.correction_interval:
                dv_correction = self._guidance(r, v)
                if np.linalg.norm(dv_correction) > 1e-8:
                    v = v + dv_correction
                    total_dv += np.linalg.norm(dv_correction)
                    corrections.append((t, dv_correction.copy()))
                time_since_correction = 0.0

            states.append(RelativeState(t=t, r=r.copy(), v=v.copy()))

            if np.linalg.norm(r) <= self.docking_threshold:
                break

        final_range = float(np.linalg.norm(r))
        success = final_range <= self.docking_threshold

        return RendezvousResult(
            states=states,
            dv_corrections=corrections,
            success=success,
            final_range=final_range,
            total_dv=total_dv,
        )

    def _guidance(self, r: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Simple proportional guidance law.

        Applies a correction burn to reduce along-track and radial
        offset. Targets a straight-line approach to the origin (docking
        port) at constant closing speed.

        Args:
            r: Current relative position (km).
            v: Current relative velocity (km/s).

        Returns:
            Delta-v correction vector (km/s).
        """
        range_mag = np.linalg.norm(r)
        if range_mag < self.docking_threshold:
            return np.zeros(3)

        # Desired velocity: approach at constant speed toward origin
        v_desired = -self.approach_velocity * r / range_mag
        return v_desired - v