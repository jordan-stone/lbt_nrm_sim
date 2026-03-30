"""
Two-panel visualization: focal plane + power spectrum.

Renders individual frames for GIF generation. Each frame shows:
  Left:  focal-plane PSF (cropped to 1.5 lambda/D_subap)
  Right: power spectrum with DC masked (cropped to show all splodges)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from .apertures import HOLE_RADIUS
from .fringes import PLATE_SCALE_MAS
from .analysis import make_power_spectrum, mask_dc, dc_mask_radius, power_spectrum_crop


def compute_crops(blue_wavelength_um, median_wavelength_um, dim=2048,
                  hole_radius=HOLE_RADIUS):
    """Compute all crop and mask parameters for a given filter.

    Parameters
    ----------
    blue_wavelength_um : float
        Shortest wavelength in the filter band (microns).
    median_wavelength_um : float
        Median wavelength (microns), used for focal plane crop.
    dim : int
        Image array dimension.
    hole_radius : float
        NRM hole radius in meters.

    Returns
    -------
    params : dict
        Keys: focal_hw, ps_crop_x, ps_crop_y, dc_radius, center.
    """
    d_subap = 2 * hole_radius
    lod_mas = (median_wavelength_um * 1e-6 / d_subap) * 206264.806 * 1000
    focal_hw = int(round(1.5 * lod_mas / PLATE_SCALE_MAS))

    dc_r = dc_mask_radius(blue_wavelength_um, dim=dim, hole_radius=hole_radius)
    ps_cx, ps_cy = power_spectrum_crop(blue_wavelength_um, dim=dim,
                                        hole_radius=hole_radius)

    return {
        'focal_hw': focal_hw,
        'ps_crop_x': ps_cx,
        'ps_crop_y': ps_cy,
        'dc_radius': dc_r,
        'center': dim // 2,
    }


def compute_stretches(psf_zero, params):
    """Compute vmin/vmax from the zero-offset (perfectly aligned) frame.

    Parameters
    ----------
    psf_zero : ndarray
        PSF at zero OPD, zero tip, zero tilt.
    params : dict
        From compute_crops().

    Returns
    -------
    stretches : dict
        Keys: focal_vmin, focal_vmax, ps_vmin, ps_vmax.
    """
    focal_vmax = psf_zero.max() * 0.5

    power, _ = make_power_spectrum(psf_zero)
    pm = mask_dc(power, radius_pix=params['dc_radius'])
    ps_vmax = pm.max() * 0.5

    return {
        'focal_vmin': 0.0,
        'focal_vmax': focal_vmax,
        'ps_vmin': 0.0,
        'ps_vmax': ps_vmax,
    }


def render_frame(psf, params, stretches, title_text='', cmap='inferno'):
    """Render a two-panel figure: focal plane + power spectrum.

    The focal plane panel is square (cropped to 1.5 lambda/D_subap).
    The power spectrum shows mostly positive x-frequencies: from
    -1 DC mask radius to the full positive extent derived from the
    mask geometry. The two panels have a width ratio of 1:1.5.

    Parameters
    ----------
    psf : ndarray
        Polychromatic PSF.
    params : dict
        From compute_crops().
    stretches : dict
        From compute_stretches().
    title_text : str
        Text for the super-title.
    cmap : str
        Matplotlib colormap name.

    Returns
    -------
    fig : matplotlib Figure
    """
    c = params['center']
    fhw = params['focal_hw']
    pcx = params['ps_crop_x']
    pcy = params['ps_crop_y']
    dc_r = params['dc_radius']

    power, _ = make_power_spectrum(psf)
    pm = mask_dc(power, radius_pix=dc_r)

    # Power spectrum crop: x from -1*dc_radius to +full extent
    ps_x_left = c - dc_r
    ps_x_right = c + pcx
    ps_slx = slice(ps_x_left, ps_x_right)
    ps_sly = slice(c - pcy, c + pcy)

    # Panel dimensions in pixels
    focal_npix = 2 * fhw                       # square
    ps_w_pix = ps_x_right - ps_x_left          # asymmetric x
    ps_h_pix = 2 * pcy                         # full y

    # Fixed width ratio 1:1.5, same height
    panel_height = 5.0  # inches
    focal_inches = panel_height  # square panel
    ps_inches = 1.5 * focal_inches

    total_w = focal_inches + ps_inches + 1.5  # gap for labels
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(total_w, panel_height + 1.0),
        gridspec_kw={'width_ratios': [1, 1.5]},
    )

    # Focal plane (square)
    f_sl = slice(c - fhw, c + fhw)
    ax1.imshow(psf[f_sl, f_sl], origin='lower',
               vmin=stretches['focal_vmin'], vmax=stretches['focal_vmax'],
               cmap=cmap)
    ax1.set_title('Focal plane')
    ax1.set_xlabel('pixels')
    ax1.set_ylabel('pixels')

    # Power spectrum (asymmetric x crop)
    ax2.imshow(pm[ps_sly, ps_slx], origin='lower',
               vmin=stretches['ps_vmin'], vmax=stretches['ps_vmax'],
               cmap=cmap, aspect='auto')
    ax2.set_title('Power spectrum')
    ax2.set_xlabel('pixels')

    if title_text:
        fig.suptitle(title_text, fontsize=13, fontweight='bold')

    fig.tight_layout()
    return fig


def frames_to_gif(frame_paths, output_path, duration_ms=100, loop=0):
    """Assemble saved frame PNGs into an animated GIF.

    Parameters
    ----------
    frame_paths : list of str
        Paths to frame PNG files, in order.
    output_path : str
        Output GIF path.
    duration_ms : int
        Duration of each frame in milliseconds.
    loop : int
        Number of loops (0 = infinite).
    """
    from PIL import Image

    frames = [Image.open(p) for p in frame_paths]
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=loop,
    )
