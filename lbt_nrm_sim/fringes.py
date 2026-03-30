"""
Monochromatic and polychromatic NRM fringe simulation for the LBT.

Phase errors (OPD, tip, tilt) are applied to the SX side of the aperture.
DX is the phase reference.

Functions can be used standalone (recomputing grids each call) or via
NRMSimulator which precomputes and caches the pupil geometry for speed.
"""

import numpy as np
from scipy.ndimage import map_coordinates

from .apertures import (
    make_pupil_grid,
    NRM_HOLE_COORDS,
    HOLE_RADIUS,
    MIRROR_OFFSET_X,
    MIRROR_SEPARATION,
)


# LMIRCam detector plate scale
PLATE_SCALE_MAS = 10.707  # mas per pixel


def jzoom(im, z):
    """Zoom an image by factor z while keeping dimensions and centering fixed.

    Used to rescale monochromatic fringes so that different wavelengths
    have the correct relative angular size before co-adding.

    Parameters
    ----------
    im : ndarray (dim, dim)
        Input image.
    z : float
        Zoom factor (>1 shrinks the pattern, <1 expands it).

    Returns
    -------
    out : ndarray (dim, dim)
        Rescaled image, same shape as input.
    """
    dim = im.shape[0]
    yy, xx = np.indices((dim, dim), dtype=float)

    yy_out = z * (yy - dim / 2) + dim / 2
    xx_out = z * (xx - dim / 2) + dim / 2

    return map_coordinates(im, [yy_out, xx_out], order=3, mode='constant')


def _build_masks(xx, yy, hole_radius=HOLE_RADIUS):
    """Build SX and DX hole masks from coordinate grids.

    Parameters
    ----------
    xx, yy : ndarray
        Pupil-plane coordinate grids in meters.
    hole_radius : float
        Radius of each NRM hole in meters.

    Returns
    -------
    sx_mask, dx_mask : ndarray (float)
        Binary masks for SX (negative x) and DX (positive x) holes.
    """
    sx = np.zeros(xx.shape, dtype=float)
    dx = np.zeros(xx.shape, dtype=float)
    for (hx, hy) in NRM_HOLE_COORDS:
        r2 = (yy - hy)**2 + (xx - hx)**2
        if hx < 0:
            sx[r2 <= hole_radius**2] = 1.0
        else:
            dx[r2 <= hole_radius**2] = 1.0
    return sx, dx


def make_nrm_fringe(delta_phase, tip_tot, tilt_tot,
                     dim=2048, pix_scale=0.0625, hole_radius=HOLE_RADIUS,
                     precomputed=None):
    """Compute a monochromatic NRM PSF with phase errors on SX side.

    Parameters
    ----------
    delta_phase : float
        OPD piston phase applied to SX, in radians.
    tip_tot : float
        Total edge-to-edge tip phase across the SX mirror, in radians.
        (phase gradient in the x direction)
    tilt_tot : float
        Total edge-to-edge tilt phase across the SX mirror, in radians.
        (phase gradient in the y direction)
    dim : int
        Array dimension.
    pix_scale : float
        Meters per pixel in pupil plane.
    hole_radius : float
        Radius of each NRM hole in meters.
    precomputed : dict or None
        If provided, should contain 'sx_mask', 'dx_mask', 'xx_sx', 'yy'
        from NRMSimulator. Skips grid/mask recomputation.

    Returns
    -------
    psf : ndarray (dim, dim)
        Monochromatic PSF (intensity).
    """
    if precomputed is not None:
        sx = precomputed['sx_mask']
        dx = precomputed['dx_mask']
        xx_sx = precomputed['xx_sx']
        yy = precomputed['yy']
    else:
        xx, yy = make_pupil_grid(dim, pix_scale)
        sx, dx = _build_masks(xx, yy, hole_radius)
        xx_sx = xx + MIRROR_OFFSET_X

    # SX side: apply piston + tip + tilt phase
    tip_phase = np.exp(1j * xx_sx * tip_tot / MIRROR_SEPARATION)
    tilt_phase = np.exp(1j * yy * tilt_tot / MIRROR_SEPARATION)
    pist_phase = np.exp(1j * delta_phase)

    pupil = sx * (pist_phase * tip_phase * tilt_phase) + dx

    # FFT to focal plane
    ef = np.fft.fftshift(np.fft.fft2(pupil))
    psf = (ef * ef.conj()).real

    return psf


def make_polychromatic_nrm(opd_um, tip_arcsec, tilt_arcsec,
                            waves_um, weights,
                            dim=2048, pix_scale=0.0625,
                            hole_radius=HOLE_RADIUS,
                            precomputed=None):
    """Compute a polychromatic NRM PSF with phase errors on SX side.

    For each wavelength, a monochromatic PSF is computed, rescaled via jzoom
    to account for the wavelength-dependent angular scale, then co-added
    with transmission weights. The final image is resampled to match the
    LMIRCam detector plate scale (10.707 mas/pix).

    Parameters
    ----------
    opd_um : float
        Optical path difference applied to SX, in microns.
    tip_arcsec : float
        Differential tip (x-direction) between SX and DX, in arcseconds.
    tilt_arcsec : float
        Differential tilt (y-direction) between SX and DX, in arcseconds.
    waves_um : array-like
        Wavelengths to sample, in microns.
    weights : array-like
        Transmission weights for each wavelength (should sum to 1).
    dim : int
        Array dimension for pupil-plane computation.
    pix_scale : float
        Meters per pixel in pupil plane.
    hole_radius : float
        Radius of each NRM hole in meters.
    precomputed : dict or None
        Precomputed grids/masks from NRMSimulator.

    Returns
    -------
    psf : ndarray (dim, dim)
        Polychromatic PSF resampled to LMIRCam plate scale.
    """
    waves_um = np.asarray(waves_um)
    weights = np.asarray(weights)

    # Convert physical quantities to phase (radians) at each wavelength
    delta_phases = 2 * np.pi * opd_um / waves_um

    # Tip/tilt: arcseconds -> total edge-to-edge phase in radians
    mirror_diam_um = MIRROR_SEPARATION * 1e6
    tip_throw_um = (tip_arcsec / 206264.806) * mirror_diam_um
    tilt_throw_um = (tilt_arcsec / 206264.806) * mirror_diam_um

    delta_tips = 2 * np.pi * tip_throw_um / waves_um
    delta_tilts = 2 * np.pi * tilt_throw_um / waves_um

    # Compute monochromatic fringes and rescale
    fringes_sum = np.zeros((dim, dim), dtype=float)
    ref_wave = waves_um[0]

    for ii, wave in enumerate(waves_um):
        mono = make_nrm_fringe(
            delta_phases[ii], delta_tips[ii], delta_tilts[ii],
            dim=dim, pix_scale=pix_scale, hole_radius=hole_radius,
            precomputed=precomputed,
        )
        rescaled = jzoom(mono, ref_wave / wave)
        fringes_sum += weights[ii] * rescaled

    # Resample to LMIRCam plate scale
    native_ps_arcsec = (ref_wave * 1e-6 / (dim * pix_scale)) * 206264.806
    native_ps_mas = native_ps_arcsec * 1000.0
    scale_factor = PLATE_SCALE_MAS / native_ps_mas

    psf = jzoom(fringes_sum, scale_factor)
    return psf


class NRMSimulator:
    """LBT NRM fringe simulator with precomputed pupil geometry.

    Precomputes coordinate grids and hole masks once at init, then
    passes them to the standalone functions for each call.

    Parameters
    ----------
    dim : int
        Array dimension (default 2048).
    pix_scale : float
        Meters per pixel in pupil plane (default 0.0625).
    hole_radius : float
        NRM hole radius in meters (default 0.392).

    Example
    -------
    >>> sim = NRMSimulator()
    >>> waves, weights, info = sample_filter('L')
    >>> psf = sim.make_polychromatic_nrm(0, 0, 0, waves, weights)
    """

    def __init__(self, dim=2048, pix_scale=0.0625, hole_radius=HOLE_RADIUS):
        self.dim = dim
        self.pix_scale = pix_scale
        self.hole_radius = hole_radius

        # Precompute grids and masks
        xx, yy = make_pupil_grid(dim, pix_scale)
        sx_mask, dx_mask = _build_masks(xx, yy, hole_radius)

        self._precomputed = {
            'sx_mask': sx_mask,
            'dx_mask': dx_mask,
            'xx_sx': xx + MIRROR_OFFSET_X,
            'yy': yy,
        }

    def make_nrm_fringe(self, delta_phase, tip_tot, tilt_tot):
        """Monochromatic NRM fringe. See standalone make_nrm_fringe."""
        return make_nrm_fringe(
            delta_phase, tip_tot, tilt_tot,
            dim=self.dim, pix_scale=self.pix_scale,
            hole_radius=self.hole_radius,
            precomputed=self._precomputed,
        )

    def make_polychromatic_nrm(self, opd_um, tip_arcsec, tilt_arcsec,
                                waves_um, weights):
        """Polychromatic NRM fringe. See standalone make_polychromatic_nrm."""
        return make_polychromatic_nrm(
            opd_um, tip_arcsec, tilt_arcsec,
            waves_um, weights,
            dim=self.dim, pix_scale=self.pix_scale,
            hole_radius=self.hole_radius,
            precomputed=self._precomputed,
        )
