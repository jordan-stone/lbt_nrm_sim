# lbt_nrm_sim

Simulator for aperture masking interferometry with the Large Binocular Telescope (LBT). Generates polychromatic fringe patterns and power spectra for the 12-hole non-redundant mask (NRM) on the LBT's dual 8.4m primary mirrors, with a 23m maximum baseline.

Designed as an operational aid for interpreting on-sky NRM data — particularly for diagnosing differential tip/tilt (image overlap) and optical path difference (fringe tracking) between the SX and DX sides.

## Installation

```bash
git clone https://github.com/jordan-stone/lbt_nrm_sim.git
cd lbt_nrm_sim
pip install -e .
```

Dependencies: numpy, scipy, matplotlib, Pillow.

## Quick start

### Generate alignment diagnostic GIFs

```bash
# All 6 GIFs: 3 sweeps (tip, tilt, OPD) × 2 filters (L-band, 3.9µm)
python scripts/make_gifs.py

# Selective generation
python scripts/make_gifs.py --filters L --sweeps opd --n_frames 30
python scripts/make_gifs.py --filters 3.9 --sweeps tip tilt --n_frames 50
```

Each GIF shows two panels side by side:
- **Left:** focal-plane polychromatic PSF
- **Right:** power spectrum with zero-spacing masked, showing the discrete baseline splodges

The stretch is fixed from the perfectly-aligned frame so that degradation with misalignment is visually obvious.

### Use as a library

```python
from lbt_nrm_sim.fringes import NRMSimulator
from lbt_nrm_sim.filters import sample_filter
from lbt_nrm_sim.analysis import make_power_spectrum, mask_dc

# Initialize (precomputes pupil geometry)
sim = NRMSimulator()

# Load filter
waves, weights, info = sample_filter('L')  # or '3.9'

# Compute a polychromatic PSF
psf = sim.make_polychromatic_nrm(
    opd_um=0.0,         # OPD in microns
    tip_arcsec=0.0,     # differential tip in arcsec
    tilt_arcsec=0.0,    # differential tilt in arcsec
    waves_um=waves,
    weights=weights,
)

# Power spectrum
power, phase = make_power_spectrum(psf)
```

Standalone functions (without precomputation) are also available:

```python
from lbt_nrm_sim.fringes import make_nrm_fringe, make_polychromatic_nrm
```

## Package structure

```
lbt_nrm_sim/
├── pyproject.toml
├── lbt_nrm_sim/
│   ├── apertures.py        # LBT pupil geometry, 12-hole NRM mask coordinates
│   ├── fringes.py          # Monochromatic & polychromatic fringe simulation
│   ├── filters.py          # Filter transmission curve loading & sampling
│   ├── analysis.py         # Power spectrum, DC masking, crop calculation
│   ├── visualization.py    # Two-panel frame rendering, GIF assembly
│   └── filters/            # Filter transmission data files
│       ├── LBT_LMIRCam_L_77K.dat
│       └── LBT_LMIRCam_N03946-4_77K.dat
└── scripts/
    └── make_gifs.py        # Command-line GIF generation
```

## Physics notes

- Phase errors (OPD, tip, tilt) are applied to the SX side; DX is the reference
- Polychromatic fringes weight each wavelength by the real filter transmission curve
- L-band: 25 wavelength samples across the 50%-transmission bandpass (3.41–3.99 µm)
- 3.9µm filter: 10 samples (3.83–4.07 µm), matching the L-band wavelength spacing
- The output PSF is resampled to the LMIRCam detector plate scale (10.707 mas/pixel)
- Power spectrum DC mask radius is derived from the sub-aperture autocorrelation width
- Power spectrum crop extents are derived from the mask geometry and shortest wavelength

## Runtime

Each polychromatic frame takes ~43s (L-band, 25 wavelengths) or ~17s (3.9µm, 10 wavelengths) on a single core. A 50-frame GIF takes ~36 min (L) or ~14 min (3.9µm). All 6 GIFs: ~2.5 hours.

## License

MIT
