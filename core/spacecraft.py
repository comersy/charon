"""Servicer spacecraft model for Charon.

Models the servicing vehicle's physical state, propellant budget,
and maneuver history throughout a mission.

All units: km, km/s, seconds, kg.
"""

from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

from .maneuver import ManeuverResult


MU_EARTH = 398600.4418  # km^3/s^2


@dataclass
class ManeuverRecord:
    """Log entry for a single executed maneuver.

    Attributes:
        t:          UTC timestamp of the burn.
        dv:         Delta-v vector applied (km/s).
        dv_scalar:  Magnitude of the burn (km/s).
        fuel_used:  Propellant consumed (kg).
        target:     Name of the target satellite (if applicable).
    """
    t: datetime
    dv: np.ndarray
    dv_scalar: float
    fuel_used: float
    target: str = ""


class Spacecraft:
    """Servicer spacecraft with propellant budget and maneuver tracking.

    Uses the Tsiolkovsky rocket equation to convert delta-v burns
    into propellant consumption:

        delta_m = m * (1 - exp(-dv / (Isp * g0)))

    Example:
        sc = Spacecraft(dry_mass=500.0, fuel_mass=2000.0, isp=310.0)
        sc.apply_maneuver(result, t=datetime.now(timezone.utc), target="SAT-01")
        print(sc.fuel_remaining)
    """

    G0 = 9.80665e-3  # Standard gravity in km/s^2

    def __init__(
        self,
        dry_mass: float,
        fuel_mass: float,
        isp: float = 310.0,
        name: str = "Servicer",
    ):
        """
        Args:
            dry_mass:  Dry mass of the spacecraft (kg).
            fuel_mass: Initial propellant mass (kg).
            isp:       Specific impulse in seconds. Default: 310 s (typical
                       bipropellant engine, e.g. Dragon's Draco thrusters).
            name:      Human-readable spacecraft identifier.
        """
        if dry_mass <= 0:
            raise ValueError(f"dry_mass must be positive, got {dry_mass}")
        if fuel_mass < 0:
            raise ValueError(f"fuel_mass must be >= 0, got {fuel_mass}")
        if isp <= 0:
            raise ValueError(f"Isp must be positive, got {isp}")

        self.name = name
        self.dry_mass = dry_mass
        self.isp = isp
        self._fuel_mass = fuel_mass
        self._initial_fuel = fuel_mass 
        self._history: list[ManeuverRecord] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def fuel_remaining(self) -> float:
        """Current propellant mass (kg)."""
        return self._fuel_mass

    @property
    def total_mass(self) -> float:
        """Current total mass: dry + remaining fuel (kg)."""
        return self.dry_mass + self._fuel_mass

    @property
    def fuel_fraction(self) -> float:
        """Fraction of fuel remaining (0.0 – 1.0)."""
        initial = sum(r.fuel_used for r in self._history) + self._fuel_mass
        return self._fuel_mass / initial if initial > 0 else 0.0

    @property
    def history(self) -> list[ManeuverRecord]:
        """Read-only list of all executed maneuvers."""
        return list(self._history)

    @property
    def total_dv_used(self) -> float:
        """Cumulative delta-v expended so far (km/s)."""
        return sum(r.dv_scalar for r in self._history)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def dv_to_fuel(self, dv: float) -> float:
        """Compute propellant mass needed for a given delta-v burn.

        Uses the Tsiolkovsky rocket equation:
            delta_m = m_total * (1 - exp(-dv / (Isp * g0)))

        Args:
            dv: Delta-v magnitude (km/s).

        Returns:
            Propellant mass consumed (kg).
        """
        return self.total_mass * (1 - np.exp(-dv / (self.isp * self.G0)))

    def max_dv(self) -> float:
        """Maximum achievable delta-v with current fuel (km/s).

        Derived from Tsiolkovsky:
            dv_max = Isp * g0 * ln(m_total / m_dry)
        """
        if self._fuel_mass <= 0:
            return 0.0
        return self.isp * self.G0 * np.log(self.total_mass / self.dry_mass)

    def can_perform(self, dv: float) -> bool:
        """Return True if the spacecraft has enough fuel for a dv burn."""
        return self.dv_to_fuel(dv) <= self._fuel_mass

    def apply_maneuver(
        self,
        maneuver: ManeuverResult,
        t: datetime,
        target: str = "",
    ) -> ManeuverRecord:
        """Execute a maneuver and update the fuel budget.

        Args:
            maneuver: Result from hohmann_transfer() or lambert_transfer().
            t:        UTC timestamp of the burn.
            target:   Name of the target satellite (for logging).

        Returns:
            ManeuverRecord logged to history.

        Raises:
            RuntimeError: If there is not enough fuel.
        """
        dv = maneuver.dv_total
        fuel_needed = self.dv_to_fuel(dv)

        if fuel_needed > self._fuel_mass:
            raise RuntimeError(
                f"[{self.name}] Insufficient fuel: need {fuel_needed:.2f} kg, "
                f"have {self._fuel_mass:.2f} kg."
            )

        self._fuel_mass -= fuel_needed

        record = ManeuverRecord(
            t=t,
            dv=maneuver.dv1 + maneuver.dv2,
            dv_scalar=dv,
            fuel_used=fuel_needed,
            target=target,
        )
        self._history.append(record)
        return record

    def apply_dv(self, dv: float, t: datetime, target: str = "") -> ManeuverRecord:
        """Apply a scalar delta-v directly (without a ManeuverResult object).

        Useful for quick calculations or when only the cost is known.

        Args:
            dv:     Delta-v magnitude (km/s).
            t:      UTC timestamp.
            target: Target label for logging.
        """
        from .maneuver import ManeuverResult
        dummy = ManeuverResult(
            dv1=np.array([dv, 0.0, 0.0]),
            dv2=np.zeros(3),
            dv_total=dv,
            tof=0.0,
            transfer_type="direct",
        )
        return self.apply_maneuver(dummy, t, target)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def status(self) -> str:
        """Human-readable status summary."""
        return (
            f"Spacecraft '{self.name}'\n"
            f"  Dry mass    : {self.dry_mass:.1f} kg\n"
            f"  Fuel        : {self._fuel_mass:.1f} kg "
            f"({self.fuel_fraction * 100:.1f}% remaining)\n"
            f"  Total mass  : {self.total_mass:.1f} kg\n"
            f"  Max dv left : {self.max_dv():.4f} km/s\n"
            f"  Burns done  : {len(self._history)}\n"
            f"  Total dv used: {self.total_dv_used:.4f} km/s"
        )

    def __repr__(self):
        return (
            f"Spacecraft(name='{self.name}', "
            f"fuel={self._fuel_mass:.1f}/{self.dry_mass:.1f} kg, "
            f"isp={self.isp} s)"
        )