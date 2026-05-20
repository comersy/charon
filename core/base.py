from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class OrbitalState:
    """Position (km) and velocity (km/s) in the ECI frame (J2000).

    Attributes:
        t: UTC timestamp of the state.
        r: Position vector [x, y, z] in km.
        v: Velocity vector [vx, vy, vz] in km/s.
    """
    t: datetime
    r: np.ndarray
    v: np.ndarray

    @property
    def altitude(self) -> float:
        """Altitude above Earth's surface in km."""
        R_EARTH = 6371.0
        return np.linalg.norm(self.r) - R_EARTH

    @property
    def speed(self) -> float:
        """Scalar speed in km/s."""
        return np.linalg.norm(self.v)

    def __repr__(self):
        return (
            f"OrbitalState(t={self.t.isoformat()}, "
            f"alt={self.altitude:.1f} km, "
            f"speed={self.speed:.3f} km/s)"
        )


class BasePropagator(ABC):
    """Abstract interface for all Charon propagators.

    To add a new propagator (e.g. RK4, J2 analytical), subclass this
    and implement propagate(). Everything else in the framework will
    work without modification.
    """

    @abstractmethod
    def propagate(self, t: datetime) -> OrbitalState:
        """Compute the orbital state at time t."""
        ...

    def propagate_many(self, times: list[datetime]) -> list[OrbitalState]:
        """Propagate over a list of timestamps."""
        return [self.propagate(t) for t in times]