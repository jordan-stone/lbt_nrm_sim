"""
Filter transmission curve loading and wavelength sampling.

Filter trace files are two-column ASCII: wavelength (nm), transmission (0-1).
We sample N wavelengths linearly spaced between the blue and red 50%-of-peak
transmission limits, and return the interpolated transmission at each sample
as weights for polychromatic fringe co-addition.
"""

import numpy as np
from pathlib import Path

FILTER_DIR = Path(__file__).parent / "filters"

FILTER_FILES = {
    "L": "LBT_LMIRCam_L_77K.dat",
    "3.9": "LBT_LMIRCam_N03946-4_77K.dat",
}


def load_filter(name):
    """Load a filter transmission curve.

    Parameters
    ----------
    name : str
        Filter name: 'L' or '3.9'.

    Returns
    -------
    wavelengths_um : ndarray
        Wavelengths in microns.
    transmission : ndarray
        Transmission values (0-1).
    """
    fname = FILTER_FILES[name]
    data = np.loadtxt(FILTER_DIR / fname)
    wavelengths_um = data[:, 0] / 1e4  # nm -> microns
    transmission = data[:, 1]
    return wavelengths_um, transmission


def sample_filter(name, n_samples=None):
    """Sample a filter at evenly-spaced wavelengths within the 50% bandpass.

    For L-band, n_samples defaults to 25.
    For the 3.9um filter, n_samples is chosen to match the L-band wavelength
    spacing (~0.0243 um).

    Parameters
    ----------
    name : str
        Filter name: 'L' or '3.9'.
    n_samples : int or None
        Number of wavelength samples. If None, uses defaults.

    Returns
    -------
    waves_um : ndarray
        Sampled wavelengths in microns.
    weights : ndarray
        Normalized transmission weights (sum to 1).
    info : dict
        Filter metadata: blue_50, red_50, median_um, bandwidth_um.
    """
    wl, tx = load_filter(name)

    # Find 50%-of-peak limits
    peak = tx.max()
    above_half = wl[tx > 0.5 * peak]
    blue_50 = above_half.min()
    red_50 = above_half.max()
    bandwidth = red_50 - blue_50
    median_um = (blue_50 + red_50) / 2.0

    # Default sample counts
    if n_samples is None:
        if name == "L":
            n_samples = 25
        else:
            # Match L-band spacing
            l_wl, l_tx = load_filter("L")
            l_peak = l_tx.max()
            l_above = l_wl[l_tx > 0.5 * l_peak]
            l_spacing = (l_above.max() - l_above.min()) / 24.0
            n_samples = max(2, int(round(bandwidth / l_spacing)))

    # Sample wavelengths
    waves_um = np.linspace(blue_50, red_50, n_samples)

    # Interpolate transmission at sample points and normalize to weights
    raw_weights = np.interp(waves_um, wl, tx)
    weights = raw_weights / raw_weights.sum()

    info = {
        "name": name,
        "blue_50": blue_50,
        "red_50": red_50,
        "median_um": median_um,
        "bandwidth_um": bandwidth,
        "n_samples": n_samples,
    }

    return waves_um, weights, info
