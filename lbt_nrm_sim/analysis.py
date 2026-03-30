"""
Fourier analysis of NRM fringe images.

Computes the 2D power spectrum (squared modulus of the FFT) and provides
DC masking for visualization of the high-frequency splodges.
"""

import numpy as np

from .apertures import NRM_HOLE_COORDS, HOLE_RADIUS
from .fringes import PLATE_SCALE_MAS


def make_power_spectrum(fringe_im):
    """Compute the 2D power spectrum of a fringe image.

    The image is shifted before FFT to center the DC component, matching
    the convention in the original code.

    Parameters
    ----------
    fringe_im : ndarray (N, N)
        Focal-plane fringe image (intensity).

    Returns
    -------
    power : ndarray (N, N)
        Power spectrum (|FFT|^2).
    phase : ndarray (N, N)
        Phase of the FFT, wrapped to [0, 2pi).
    """
    ft = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(fringe_im)))
    power = np.abs(ft)**2
    phase = (2 * np.pi + np.angle(ft)) % (2 * np.pi)
    return power, phase


def mask_dc(power, radius_pix=5):
    """Zero out the central DC peak in a power spectrum.

    Parameters
    ----------
    power : ndarray (N, N)
        Power spectrum.
    radius_pix : int
        Radius in pixels of the circular mask centered on DC.

    Returns
    -------
    masked : ndarray (N, N)
        Power spectrum with DC region set to zero.
    """
    dim = power.shape[0]
    c = dim // 2
    yy, xx = np.indices(power.shape)
    r2 = (yy - c)**2 + (xx - c)**2
    masked = power.copy()
    masked[r2 <= radius_pix**2] = 0.0
    return masked


def dc_mask_radius(wavelength_um, dim=2048, hole_radius=HOLE_RADIUS):
    """Compute the DC mask radius from the sub-aperture autocorrelation.

    The zero-spacing peak in the power spectrum has a width set by the
    autocorrelation of a single NRM hole (diameter = 2 * hole_radius).
    Use the shortest wavelength in the band for the most conservative mask.

    Parameters
    ----------
    wavelength_um : float
        Shortest wavelength in the band, in microns.
    dim : int
        Image array dimension.
    hole_radius : float
        NRM hole radius in meters.

    Returns
    -------
    radius_pix : int
        Recommended DC mask radius in power spectrum pixels.
    """
    lam_m = wavelength_um * 1e-6
    ps_rad = PLATE_SCALE_MAS * 1e-3 / 206264.806
    d_hole = 2 * hole_radius
    r = d_hole * dim * ps_rad / lam_m
    return int(np.ceil(r))


def power_spectrum_crop(wavelength_um, dim=2048, hole_radius=HOLE_RADIUS,
                        margin=1.1):
    """Compute crop extents to show all baseline splodges in the power spectrum.

    Each splodge is centered at a position set by the baseline length and
    has a width set by the sub-aperture autocorrelation. The crop is sized
    to include the outermost splodge edges plus a margin. Use the shortest
    wavelength in the band — splodges are widest and most spread out there.

    Parameters
    ----------
    wavelength_um : float
        Shortest wavelength in the band, in microns.
    dim : int
        Image array dimension.
    hole_radius : float
        NRM hole radius in meters.
    margin : float
        Fractional margin beyond outermost splodge edge (default 1.1 = 10%).

    Returns
    -------
    crop_x, crop_y : int
        Half-widths in pixels for the power spectrum crop.
    """
    coords = np.array(NRM_HOLE_COORDS)
    lam_m = wavelength_um * 1e-6
    ps_rad = PLATE_SCALE_MAS * 1e-3 / 206264.806
    d_hole = 2 * hole_radius

    # Find longest baseline projections in x and y
    max_bx = 0.0
    max_by = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            bx = abs(coords[j, 0] - coords[i, 0])
            by = abs(coords[j, 1] - coords[i, 1])
            max_bx = max(max_bx, bx)
            max_by = max(max_by, by)

    # Outermost splodge edge = baseline + hole radius (autocorrelation half-width)
    outer_x = (max_bx + d_hole) * dim * ps_rad / lam_m
    outer_y = (max_by + d_hole) * dim * ps_rad / lam_m

    crop_x = int(np.ceil(outer_x * margin))
    crop_y = int(np.ceil(outer_y * margin))

    return crop_x, crop_y
