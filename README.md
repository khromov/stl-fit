# STL Build Volume Fit Checker

Determine if an STL model fits within a 3D printer build volume in any orientation by searching over all possible rotations.

## How It Works

This tool solves the optimal rotation / minimum bounding box problem for 3D printing. Given an STL file and a build volume, it finds the best orientation to fit the model, or reports if it's impossible.

**Key optimizations:**

1. **Convex hull reduction** — Only the convex hull vertices matter for computing axis-aligned bounding boxes. This reduces the working set from millions of vertices to thousands, making the search tractable.

2. **Uniform SO(3) sampling** — Generates 500,000 uniformly distributed 3D rotations using `scipy.spatial.transform.Rotation.random()`, avoiding the gimbal lock and bias issues of naive Euler angle sweeps.

3. **Vectorized projection method** — Computes AABB extents for thousands of rotations in parallel using optimized NumPy matrix operations, avoiding memory-intensive tensor allocations.

4. **Multi-start local refinement** — Refines the top-5 coarse results using Nelder-Mead optimization to find the global optimum orientation.

**Performance:** Processes 500,000 rotations in ~30 seconds for large medical scan models (1M+ triangles).

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `trimesh`

## Usage

### Basic Usage

Check if a model fits in the default 180×180×180mm build volume:

```bash
python stl_fit.py --stl model.stl
```

### Custom Build Volume

Specify custom dimensions in millimeters (X Y Z):

```bash
python stl_fit.py --stl model.stl --build-volume 200 200 300
```

### Reproducible Results

Use a random seed for deterministic results:

```bash
python stl_fit.py --stl model.stl --seed 42
```

### Faster Search

Reduce the number of sampled rotations for quicker (but less thorough) search:

```bash
python stl_fit.py --stl model.stl --samples 100000
```

### Export Rotated Model

Save optimally rotated models as new STL files. When the model fits, the tool automatically exports **5 diverse orientations** to give you printing options:

```bash
python stl_fit.py --stl model.stl --output rotated.stl
```

This creates:
- `rotated_1.stl` — Best orientation (minimizes bounding box)
- `rotated_2.stl` — Maximally different orientation (~180° rotation)
- `rotated_3.stl` — Another diverse orientation
- `rotated_4.stl` — Another diverse orientation
- `rotated_5.stl` — Another diverse orientation

**Why 5 orientations?** Different rotations offer different printing characteristics (support requirements, overhang angles, layer adhesion, etc.). The tool finds orientations that are as different as possible while still fitting, giving you practical choices for your specific printer and material.

### Full Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--stl` | path | *(required)* | Path to STL file |
| `--build-volume X Y Z` | float × 3 | `180 180 180` | Build volume dimensions in mm |
| `--samples` | int | `500000` | Number of random rotations to test |
| `--batch-size` | int | `5000` | Rotations processed per batch (auto-adjusted for memory) |
| `--seed` | int | *(none)* | Random seed for reproducibility |
| `--output` | path | *(none)* | Output path for rotated STL file |

## Example Output

### When the model fits:

```
Build volume:      180.0 x 180.0 x 180.0 mm
Loading STL: MedicalScan_Skull_TN.stl
  Triangles:       1,876,176
  Vertices:        938,071
  Original AABB:   144.7 x 215.5 x 161.3 mm
  Convex hull:     41,134 vertices

Coarse search: 500,000 random rotations (batch size 1,012)
  [100%] 495/495 batches (32.4s) | Best score: 0.9727
  Best score: 0.9727 (fit found!)
  Found 982 fitting orientations in coarse search

Refining from top 5 candidates...
  Start 1: coarse 0.9727 -> refined 0.9702
  Start 2: coarse 0.9745 -> refined 0.9702
  Start 3: coarse 0.9833 -> refined 0.9702
  Start 4: coarse 0.9875 -> refined 0.9709
  Start 5: coarse 0.9890 -> refined 0.9749
  Best refined score: 0.9702

============================================================
  RESULT: FIT
============================================================
  Bounding box:  174.6 x 174.6 x 174.6 mm
                 (sorted: 174.6, 174.6, 174.6)
  Build volume:  180.0 x 180.0 x 180.0 mm
                 (sorted: 180.0, 180.0, 180.0)
  Margins:       +5.4, +5.4, +5.4 mm

  Rotation (Euler ZYX): yaw=-89.5°, pitch=80.2°, roll=-46.4°
  Rotation (rotvec):    [-1.657, 0.546, -1.775]
============================================================
```

### With `--output` flag (exports 5 diverse orientations):

When you specify `--output`, the tool exports 5 maximally different orientations:

```
Refining all 982 fitting candidates...
  [100%] 982/982 refined, 982 still fit
  982 refined orientations still fit after optimization

Selecting 5 maximally diverse orientations from 982 candidates...
  Solution 2: distance from nearest = 3.142 rad (180.0°)
  Solution 3: distance from nearest = 3.142 rad (180.0°)
  Solution 4: distance from nearest = 3.142 rad (180.0°)
  Solution 5: distance from nearest = 2.094 rad (120.0°)

============================================================
  EXPORTING 5 DIVERSE ORIENTATIONS
============================================================

Solution 1:
  File:      rotated_1.stl
  AABB:      174.6 x 174.6 x 174.6 mm
  Margins:   +5.4, +5.4, +5.4 mm
  Euler:     yaw=43.6°, pitch=6.8°, roll=97.2°
  Distance:  0.000 rad (0.0° from solution 1)

Solution 2:
  File:      rotated_2.stl
  AABB:      174.6 x 174.6 x 174.6 mm
  Margins:   +5.4, +5.4, +5.4 mm
  Euler:     yaw=-136.4°, pitch=-6.8°, roll=82.8°
  Distance:  3.142 rad (180.0° from solution 1)

... (solutions 3-5 omitted for brevity)

============================================================
  Saved 5 rotated STL files successfully!
============================================================
```

### When the model doesn't fit:

```
============================================================
  RESULT: DOES NOT FIT
============================================================
  Best bounding box: 182.1 x 179.4 x 176.8 mm
  Build volume:      180.0 x 180.0 x 180.0 mm
  Excess:            +2.1, 0.0, 0.0 mm  (exceeds by 2.1mm on one axis)
  Score:             1.0117 (needs 1.17% reduction to fit)

  Closest rotation (Euler ZYX): yaw=34.2°, pitch=-12.7°, roll=67.1°
  Rotation (rotvec):    [0.234, -1.023, 0.871]
============================================================
```

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
