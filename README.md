# stl-fit - STL orientation optimizer

Get the most out of your 3D printer. Print models that weren't possible before!

![](./.github/orientation.png)

## How It Works

This tool solves the optimal rotation / minimum bounding box problem for 3D printing. Given an STL file and a build volume, it finds the best orientation to fit the model, or reports if it's impossible.

The script is configured for a build volume of 180x180x180 by default (size of A1 mini, which often lacks presets for larger models on Makerworld)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Check if a model fits in the default 180×180×180mm build volume:

```bash
python3 stl_fit.py --stl model.stl
```

### Custom Build Volume

Specify a different cubic build volume size in millimeters:

```bash
python3 stl_fit.py --stl model.stl --build-volume 170
```

### Reproducible Results

Use a random seed for deterministic results:

```bash
python3 stl_fit.py --stl model.stl --seed 42
```

### Faster Search

Reduce the number of sampled rotations for quicker (but less thorough) search:

```bash
python3 stl_fit.py --stl model.stl --samples 100000
```

### Automatic Export of Rotated Models

When the model fits, the tool **automatically exports 10 diverse orientations** as new STL files:

```bash
python3 stl_fit.py --stl MedicalScan_Skull_TN.stl
```

This creates:

- `MedicalScan_Skull_TN_rotated_1.stl` — Best orientation (minimizes bounding box)
- `MedicalScan_Skull_TN_rotated_2.stl` — Maximally different orientation (~180° rotation)
- `MedicalScan_Skull_TN_rotated_3.stl` through `_10.stl` — Other diverse orientations

**Customize the output name** with `--output`:

```bash
python3 stl_fit.py --stl model.stl --output custom.stl
# Creates: custom_1.stl through custom_10.stl
```

**Why 10 orientations?** Different rotations offer different printing characteristics (support requirements, overhang angles, layer adhesion, etc.). The tool finds orientations that are as different as possible while still fitting, giving you practical choices for your specific printer and material.

### Full Options

| Option           | Type  | Default      | Description                                              |
| ---------------- | ----- | ------------ | -------------------------------------------------------- |
| `--stl`          | path  | _(required)_ | Path to STL file                                         |
| `--build-volume` | float | `180`        | Build volume cube size in mm                             |
| `--samples`      | int   | `500000`     | Number of random rotations to test                       |
| `--batch-size`   | int   | `5000`       | Rotations processed per batch (auto-adjusted for memory) |
| `--seed`         | int   | _(none)_     | Random seed for reproducibility                          |
| `--output`       | path  | _(auto)_     | Output path for rotated STL files (auto-generated)       |

## HTTP API

A Flask API is also available for programmatic access:

```bash
python3 api.py  # starts on http://localhost:8000
```

Upload an STL and get back the optimally rotated version:

```bash
curl -X POST -F "file=@MedicalScan_Skull_TN.stl" -o rotated.stl http://localhost:8000/optimize
```

Optional form fields: `build_volume` (default 180), `samples` (default 500000), `seed`.

Response headers include `X-STL-Fit-Score`, `X-STL-Fit-Scale-Percent`, and `X-STL-Fit-Extents`.

## Testing

```bash
pip install pytest
python3 -m pytest test_stl_fit.py -v
```

No external STL files are needed — test fixtures generate geometry programmatically.

## Exit Codes

- **0** — Model fits in the build volume
- **1** — Model does NOT fit in any orientation
- **2** — Error (file not found, invalid STL, missing dependencies, etc.)

## Why This Tool?

Slicer software like Bambu Studio, PrusaSlicer, and Cura have auto-orient features, but they optimize for **printability** (minimizing overhangs, support material) not for **fitting** within the build volume. They won't help if your model is simply too large.

This tool answers a different question: **"Can this model physically fit on my printer at all?"** — and if so, what's the optimal rotation?

## Algorithm Details

1. **Load & Reduce** — Load the STL and compute its convex hull, reducing vertices by 100-1000×
2. **Coarse Search** — Sample 500K uniformly distributed 3D rotations, compute AABB for each using vectorized projections
3. **Refinement** — Take the top-5 candidates and optimize them locally using Nelder-Mead (rotation vector parameterization to avoid gimbal lock)
4. **Report** — Output the best fit with rotation angles, margins/excess, and exit with appropriate code

The sorted-dimension comparison correctly solves the axis assignment problem — which model dimension fits in which printer dimension — without combinatorial search.

## License

MIT
