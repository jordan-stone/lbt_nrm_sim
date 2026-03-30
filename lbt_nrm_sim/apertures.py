"""
LBT aperture geometry: binocular filled apertures and 12-hole NRM mask.

Coordinate system:
    - x is along the baseline direction (separating SX and DX mirrors)
    - y is perpendicular to the baseline
    - Origin is at the center of the binocular pupil
    - Units are meters

LBT parameters:
    - Primary mirror diameter: 8.25 m (effective)
    - Primary mirror center-to-center separation: 14.416 m
    - Mirror center offset from origin: ±7.208 m in x
"""

import numpy as np

# LBT geometry (meters)
MIRROR_RADIUS = 4.125          # effective radius of each primary
MIRROR_OFFSET_X = 7.208        # center of each mirror from origin
MIRROR_SEPARATION = 8.25       # edge-to-edge, used for phase normalization

# NRM hole coordinates (x, y) in meters, in the full binocular pupil.
# 6 holes on each side (SX at negative x, DX at positive x).
NRM_HOLE_COORDS = [
    ( 5.62600,  3.15868),
    (-4.21950, -1.71358),
    ( 4.92275,  1.94061),
    (-4.92275, -0.495515),
    ( 4.21950, -1.71358),
    (-4.92275,  1.94061),
    (-9.14225, -2.93164),
    ( 9.14225, -2.93164),
    (-10.5487, -0.495515),
    ( 10.5487, -0.495515),
    (-7.03250,  3.15868),
    ( 7.03250,  3.15868),
]

# Default NRM hole radius (meters). The variable in the original code was
# called ap_diam but was used as a radius in the circle test r² <= val².
HOLE_RADIUS = 0.392


def make_pupil_grid(dim=2048, pix_scale=0.0625):
    """Create a pupil-plane coordinate grid.

    Parameters
    ----------
    dim : int
        Array size (pixels, square).
    pix_scale : float
        Meters per pixel.

    Returns
    -------
    xx, yy : ndarray
        2D coordinate arrays in meters, centered on origin.
    """
    yy, xx = np.indices((dim, dim), dtype=float)
    yy -= dim / 2
    xx -= dim / 2
    yy *= pix_scale
    xx *= pix_scale
    return xx, yy


def make_filled_aperture(dim=2048, pix_scale=0.0625):
    """Two filled circular apertures (SX + DX) of the LBT.

    Returns
    -------
    ap : ndarray (dim, dim)
        Binary pupil mask, 1 inside apertures.
    """
    xx, yy = make_pupil_grid(dim, pix_scale)

    sx = ((yy)**2 + (xx - MIRROR_OFFSET_X)**2) <= MIRROR_RADIUS**2
    dx = ((yy)**2 + (xx + MIRROR_OFFSET_X)**2) <= MIRROR_RADIUS**2

    return (sx | dx).astype(float)


def make_nrm_aperture(dim=2048, pix_scale=0.0625, hole_radius=HOLE_RADIUS):
    """12-hole non-redundant mask in the LBT binocular pupil.

    Parameters
    ----------
    dim : int
        Array size.
    pix_scale : float
        Meters per pixel.
    hole_radius : float
        Radius of each mask hole in meters.

    Returns
    -------
    ap : ndarray (dim, dim)
        Binary mask, 1 inside holes.
    """
    xx, yy = make_pupil_grid(dim, pix_scale)

    ap = np.zeros((dim, dim), dtype=float)
    for (hx, hy) in NRM_HOLE_COORDS:
        ap[((yy - hy)**2 + (xx - hx)**2) <= hole_radius**2] = 1.0

    return ap


def classify_holes():
    """Return indices of SX (negative x) and DX (positive x) holes.

    Returns
    -------
    sx_indices, dx_indices : list of int
        Indices into NRM_HOLE_COORDS for each side.
    """
    sx = [i for i, (hx, _) in enumerate(NRM_HOLE_COORDS) if hx < 0]
    dx = [i for i, (hx, _) in enumerate(NRM_HOLE_COORDS) if hx >= 0]
    return sx, dx
