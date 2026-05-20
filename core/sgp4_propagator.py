from datetime import datetime
import numpy as np
from sgp4.api import Satrec, jday

from .base import BasePropagator, OrbitalState


_SGP4_ERRORS = {
    1: "Mean motion is less than zero",
    2: "Eccentricity out of range (e >= 1 or e < -0.001)",
    3: "Satellite has decayed (sub-orbital)",
    4: "Satellite too far from Earth (hyperbolic orbit)",
    5: "Epoch too far from current time",
    6: "SGP4 element saturation",
}


class SGP4Propagator(BasePropagator):
    """SGP4/SDP4 propagator (NORAD standard).

    Accepts a Two-Line Element set (TLE) and computes position and
    velocity in the TEME frame. For Charon's purposes, TEME is treated
    as equivalent to ECI (J2000) — the error is negligible for mission
    planning at this fidelity level.

    Example:
        tle = '''ISS (ZARYA)
        1 25544U 98067A   24001.50000000  .00001234  00000+0  12345-4 0  9990
        2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815764429440'''
        prop = SGP4Propagator.from_tle_string(tle)
        state = prop.propagate(datetime.now(timezone.utc))
    """

    def __init__(self, tle_line1: str, tle_line2: str, name: str = ""):
        self.name = name
        self.tle_line1 = tle_line1
        self.tle_line2 = tle_line2
        self._sat = Satrec.twoline2rv(tle_line1, tle_line2)

    @classmethod
    def from_tle_string(cls, tle_block: str) -> "SGP4Propagator":
        """Parse a 2- or 3-line TLE block (with optional name line)."""
        lines = [l.strip() for l in tle_block.strip().splitlines() if l.strip()]
        if len(lines) == 3:
            name, line1, line2 = lines
        elif len(lines) == 2:
            name, line1, line2 = "", lines[0], lines[1]
        else:
            raise ValueError(f"Invalid TLE format: expected 2 or 3 lines, got {len(lines)}")
        return cls(line1, line2, name=name)

    def propagate(self, t: datetime) -> OrbitalState:
        """Compute orbital state at time t.

        Args:
            t: UTC datetime (must be timezone-aware).

        Returns:
            OrbitalState with position (km) and velocity (km/s).

        Raises:
            ValueError: If t is not timezone-aware.
            RuntimeError: If SGP4 returns a non-zero error code.
        """
        if t.tzinfo is None:
            raise ValueError("Datetime must be timezone-aware (UTC).")

        jd, fr = jday(t.year, t.month, t.day,
                      t.hour, t.minute, t.second + t.microsecond / 1e6)
        err, r, v = self._sat.sgp4(jd, fr)

        if err != 0:
            msg = _SGP4_ERRORS.get(err, f"Unknown SGP4 error (code {err})")
            raise RuntimeError(f"SGP4 [{self.name}]: {msg}")

        return OrbitalState(t=t, r=np.array(r), v=np.array(v))

    def __repr__(self):
        return f"SGP4Propagator(name='{self.name}', norad={self._sat.satnum})"