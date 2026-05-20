"""Orbital maneuver computations for Charon.

Provides delta-v calculations for:
- Hohmann transfers (coplanar circular orbits, minimum energy)
- Lambert transfers (general case: two positions + transfer time)

All units: km, km/s, seconds.
"""

from dataclasses import dataclass
import numpy as np


MU_EARTH = 398600.4418  # Earth gravitational parameter (km^3/s^2)


@dataclass
class ManeuverResult:
    """Result of a delta-v calculation between two orbital states.

    Attributes:
        dv1:           First burn delta-v vector (km/s). Applied at departure.
        dv2:           Second burn delta-v vector (km/s). Applied at arrival.
        dv_total:      Total scalar delta-v cost (km/s).
        tof:           Time of flight in seconds.
        transfer_type: Human-readable label ('hohmann' or 'lambert').
    """
    dv1: np.ndarray
    dv2: np.ndarray
    dv_total: float
    tof: float
    transfer_type: str

    def __repr__(self):
        return (
            f"ManeuverResult(type={self.transfer_type}, "
            f"dv_total={self.dv_total:.4f} km/s, "
            f"tof={self.tof / 3600:.2f} h)"
        )


def hohmann_transfer(r1: float, r2: float) -> ManeuverResult:
    """Compute a Hohmann transfer between two coplanar circular orbits.

    This is the minimum-energy two-burn transfer. Used as a fast
    approximation when both orbits are roughly circular and coplanar.

    Args:
        r1: Radius of departure orbit (km from Earth center).
        r2: Radius of arrival orbit (km from Earth center).

    Returns:
        ManeuverResult with scalar dv1, dv2 along the velocity vector.

    Reference:
        Curtis, H. (2013). Orbital Mechanics for Engineering Students.
        Chapter 6.
    """
    v1 = np.sqrt(MU_EARTH / r1)
    v2 = np.sqrt(MU_EARTH / r2)
    a_transfer = (r1 + r2) / 2.0

    v_perigee = np.sqrt(MU_EARTH * (2 / r1 - 1 / a_transfer))
    v_apogee  = np.sqrt(MU_EARTH * (2 / r2 - 1 / a_transfer))

    dv1_scalar = abs(v_perigee - v1)
    dv2_scalar = abs(v2 - v_apogee)

    tof = np.pi * np.sqrt(a_transfer ** 3 / MU_EARTH)

    dv1_vec = np.array([dv1_scalar, 0.0, 0.0])
    dv2_vec = np.array([dv2_scalar, 0.0, 0.0])

    return ManeuverResult(
        dv1=dv1_vec,
        dv2=dv2_vec,
        dv_total=dv1_scalar + dv2_scalar,
        tof=tof,
        transfer_type="hohmann",
    )


def lambert_transfer(
    r1: np.ndarray,
    r2: np.ndarray,
    tof: float,
    mu: float = MU_EARTH,
    prograde: bool = True,
    max_iter: int = 50,
    tol: float = 1e-10,
) -> ManeuverResult:
    """Solve Lambert's problem: find the transfer orbit between two position
    vectors in a given time of flight.

    Uses Izzo's universal variable method (robust, handles all transfer types).

    Args:
        r1:       Departure position vector (km).
        r2:       Arrival position vector (km).
        tof:      Time of flight in seconds.
        mu:       Gravitational parameter (km^3/s^2). Defaults to Earth.
        prograde: If True, enforce prograde (short-way) transfer.
        max_iter: Maximum iterations for the root-finding loop.
        tol:      Convergence tolerance.

    Returns:
        ManeuverResult with dv1 (departure burn) and dv2 (arrival burn).

    Raises:
        ValueError: If tof <= 0 or if the solver fails to converge.

    Reference:
        Izzo, D. (2015). Revisiting Lambert's problem.
        Celestial Mechanics and Dynamical Astronomy, 121(1), 1-15.
    """
    if tof <= 0:
        raise ValueError(f"Time of flight must be positive, got {tof} s")

    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)

    cos_dnu = np.dot(r1, r2) / (r1_norm * r2_norm)
    cos_dnu = np.clip(cos_dnu, -1.0, 1.0)

    cross = np.cross(r1, r2)
    if prograde:
        dnu = np.arccos(cos_dnu) if cross[2] >= 0 else (2 * np.pi - np.arccos(cos_dnu))
    else:
        dnu = (2 * np.pi - np.arccos(cos_dnu)) if cross[2] >= 0 else np.arccos(cos_dnu)

    A = np.sin(dnu) * np.sqrt(r1_norm * r2_norm / (1 - np.cos(dnu)))

    if abs(A) < 1e-10:
        raise ValueError("Degenerate Lambert geometry (collinear vectors).")

    z = 0.0
    for _ in range(max_iter):
        S, C = _stumpff(z)
        y = r1_norm + r2_norm + A * (z * S - 1) / np.sqrt(C)
        if A > 0 and y < 0:
            z += 0.1
            continue

        x = np.sqrt(y / C)
        t_computed = (x ** 3 * S + A * np.sqrt(y)) / np.sqrt(mu)

        if abs(z) > 1e-6:
            dS, dC = _stumpff_deriv(z, S, C)
            dtdz = (
                x ** 3 * (dS - 3 * S * dC / (2 * C))
                + A / 8 * (3 * S * np.sqrt(y) / C + A / x)
            ) / np.sqrt(mu)
        else:
            dtdz = np.sqrt(2) / 40 * y ** 1.5 / np.sqrt(mu)

        dz = (tof - t_computed) / dtdz
        z += dz

        if abs(dz) < tol:
            break
    else:
        raise ValueError(f"Lambert solver did not converge in {max_iter} iterations.")

    S, C = _stumpff(z)
    y = r1_norm + r2_norm + A * (z * S - 1) / np.sqrt(C)

    f  = 1 - y / r1_norm
    g  = A * np.sqrt(y / mu)
    dg = 1 - y / r2_norm

    v1 = (r2 - f * r1) / g
    v2 = (dg * r2 - r1) / g

    v1_circ = np.sqrt(mu / r1_norm) * np.array([-r1[1], r1[0], 0]) / r1_norm
    v2_circ = np.sqrt(mu / r2_norm) * np.array([-r2[1], r2[0], 0]) / r2_norm

    dv1 = v1 - v1_circ
    dv2 = v2_circ - v2

    return ManeuverResult(
        dv1=dv1,
        dv2=dv2,
        dv_total=np.linalg.norm(dv1) + np.linalg.norm(dv2),
        tof=tof,
        transfer_type="lambert",
    )


def _stumpff(z: float) -> tuple[float, float]:
    """Stumpff functions S(z) and C(z) for universal variable formulation."""
    if z > 1e-6:
        sq = np.sqrt(z)
        return (sq - np.sin(sq)) / (sq ** 3), (1 - np.cos(sq)) / z
    elif z < -1e-6:
        sq = np.sqrt(-z)
        return (np.sinh(sq) - sq) / (sq ** 3), (1 - np.cosh(sq)) / z
    else:
        return 1/6, 1/2


def _stumpff_deriv(z: float, S: float, C: float) -> tuple[float, float]:
    """Derivatives of Stumpff functions dS/dz and dC/dz."""
    if abs(z) < 1e-6:
        return -1/60, -1/24
    dS = (C - 3 * S) / (2 * z)
    dC = (1 - z * S - 2 * C) / (2 * z)
    return dS, dC