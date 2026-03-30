#!/usr/bin/env python3
"""
Generate 6 two-panel GIFs showing NRM fringe sensitivity to alignment errors.

3 sweeps (tip, tilt, OPD) x 2 filters (L-band, 3.9um) = 6 GIFs.

Each GIF has two panels:
  Left:  focal-plane polychromatic PSF
  Right: power spectrum (DC masked, linear stretch)

Usage:
    python make_gifs.py [--n_frames 50] [--outdir ./gifs]
"""

import argparse
import os
import time
import numpy as np

from lbt_nrm_sim.fringes import NRMSimulator, PLATE_SCALE_MAS
from lbt_nrm_sim.filters import sample_filter
from lbt_nrm_sim.apertures import HOLE_RADIUS
from lbt_nrm_sim.visualization import (compute_crops, compute_stretches,
                                        render_frame, frames_to_gif)


# LBT primary mirror diameter (meters) — for lambda/D tip/tilt range
D_PRIMARY = 8.4


def sweep_values(half_range, n_frames):
    """Generate parameter values sweeping from -half_range to +half_range.

    Values go negative -> zero -> positive, with zero always included.

    Parameters
    ----------
    half_range : float
        Maximum absolute value of the sweep.
    n_frames : int
        Total number of frames.

    Returns
    -------
    values : ndarray
    """
    return np.linspace(-half_range, half_range, n_frames)


def make_one_gif(sim, waves, weights, filter_info, sweep_type, n_frames,
                 outdir, frame_duration_ms=100):
    """Generate a single two-panel GIF.

    Parameters
    ----------
    sim : NRMSimulator
    waves, weights : arrays from sample_filter
    filter_info : dict from sample_filter
    sweep_type : str, one of 'tip', 'tilt', 'opd'
    n_frames : int
    outdir : str
    frame_duration_ms : int

    Returns
    -------
    gif_path : str
    """
    filter_name = filter_info['name']
    median_lam = filter_info['median_um']
    blue_lam = filter_info['blue_50']

    # Compute sweep range
    if sweep_type in ('tip', 'tilt'):
        # ±2 lambda/D of the 8.4m primary, in arcseconds
        lod_arcsec = (median_lam * 1e-6 / D_PRIMARY) * 206264.806
        half_range = 2 * lod_arcsec
        unit = 'arcsec'
        unit_symbol = '"'
    else:  # opd
        # ±2 lambda in microns
        half_range = 2 * median_lam
        unit = 'um'
        unit_symbol = ' µm'

    values = sweep_values(half_range, n_frames)

    # Compute crop/mask parameters from shortest wavelength
    params = compute_crops(blue_lam, median_lam)

    # Compute stretches from the zero-offset frame
    print(f"  Computing zero-offset reference frame...")
    t0 = time.time()
    psf_zero = sim.make_polychromatic_nrm(0, 0, 0, waves, weights)
    print(f"  Reference frame: {time.time()-t0:.1f}s")

    stretches = compute_stretches(psf_zero, params)

    # Generate frames
    frame_dir = os.path.join(outdir, f'frames_{filter_name}_{sweep_type}')
    os.makedirs(frame_dir, exist_ok=True)
    frame_paths = []

    total_t0 = time.time()
    for ii, val in enumerate(values):
        if sweep_type == 'tip':
            opd, tip, tilt = 0.0, val, 0.0
            label = f'tip = {val:+.4f}{unit_symbol}'
        elif sweep_type == 'tilt':
            opd, tip, tilt = 0.0, 0.0, val
            label = f'tilt = {val:+.4f}{unit_symbol}'
        else:
            opd, tip, tilt = val, 0.0, 0.0
            label = f'OPD = {val:+.3f}{unit_symbol}'

        t0 = time.time()
        psf = sim.make_polychromatic_nrm(opd, tip, tilt, waves, weights)
        dt = time.time() - t0

        title = f'{filter_name}-band  |  {label}  |  λ_med = {median_lam:.3f} µm'
        fig = render_frame(psf, params, stretches, title_text=title)

        fpath = os.path.join(frame_dir, f'frame_{ii:04d}.png')
        fig.savefig(fpath, dpi=100)
        import matplotlib.pyplot as plt
        plt.close(fig)
        frame_paths.append(fpath)

        elapsed = time.time() - total_t0
        eta = elapsed / (ii + 1) * (n_frames - ii - 1)
        print(f"  [{ii+1}/{n_frames}] {label}  ({dt:.1f}s, ETA {eta/60:.1f}m)")

    # Assemble GIF
    gif_path = os.path.join(outdir, f'nrm_{filter_name}_{sweep_type}.gif')
    frames_to_gif(frame_paths, gif_path, duration_ms=frame_duration_ms)
    total_dt = time.time() - total_t0
    print(f"  Saved {gif_path} ({total_dt/60:.1f} min total)")
    return gif_path


def main():
    parser = argparse.ArgumentParser(description='Generate LBT NRM alignment GIFs')
    parser.add_argument('--n_frames', type=int, default=50,
                        help='Number of frames per GIF (default: 50)')
    parser.add_argument('--outdir', type=str, default='./gifs',
                        help='Output directory (default: ./gifs)')
    parser.add_argument('--frame_duration', type=int, default=100,
                        help='Frame duration in ms (default: 100)')
    parser.add_argument('--filters', nargs='+', default=['L', '3.9'],
                        choices=['L', '3.9'],
                        help='Which filters to generate (default: both)')
    parser.add_argument('--sweeps', nargs='+', default=['tip', 'tilt', 'opd'],
                        choices=['tip', 'tilt', 'opd'],
                        help='Which sweeps to generate (default: all)')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Initialize simulator (precomputes masks)
    print("Initializing NRM simulator...")
    sim = NRMSimulator()
    print("Done.\n")

    for filt_name in args.filters:
        waves, weights, info = sample_filter(filt_name)
        print(f"=== Filter: {filt_name} ===")
        print(f"  {info['n_samples']} wavelengths: "
              f"{info['blue_50']:.3f} - {info['red_50']:.3f} µm")
        print(f"  Median: {info['median_um']:.3f} µm\n")

        for sweep in args.sweeps:
            print(f"--- Generating {sweep} sweep ---")
            make_one_gif(sim, waves, weights, info, sweep,
                         n_frames=args.n_frames, outdir=args.outdir,
                         frame_duration_ms=args.frame_duration)
            print()

    print("All done!")


if __name__ == '__main__':
    main()
