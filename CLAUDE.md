# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**stl-fit** is a Python CLI tool that determines if a 3D model (STL file) can fit within a 3D printer's build volume in any orientation. It solves the optimal rotation / minimum bounding box problem using randomized search over SO(3) (3D rotation space) followed by gradient-based local refinement.

**Exit codes:**
- 0: Model fits in build volume
- 1: Model does NOT fit in any orientation
- 2: Error (file not found, invalid STL, missing dependencies)

## Commands

### Setup
```bash
pip install -r requirements.txt
```

### Running the Tool

Basic usage (default 180×180×180mm build volume):
```bash
python3 stl_fit.py --stl model.stl
```

Custom build volume size:
```bash
python3 stl_fit.py --stl model.stl --build-volume 170
```

Faster search (fewer samples):
```bash
python3 stl_fit.py --stl model.stl --samples 100000
```

Reproducible results:
```bash
python3 stl_fit.py --stl model.stl --seed 42
```

Custom output path:
```bash
python3 stl_fit.py --stl model.stl --output custom.stl
```

## Architecture

### Core Algorithm Flow

The tool uses a two-phase optimization approach:

1. **Load & Prepare** (`load_and_prepare`)
   - Load STL mesh using trimesh
   - Handle multi-body STL files (Scene objects)
   - Compute convex hull to reduce vertex count (100-1000× reduction)
   - Center mesh and hull at origin for rotation stability

2. **Coarse Search** (`coarse_search`)
   - Generate uniform random rotations over SO(3) using `scipy.spatial.transform.Rotation.random()`
   - Process rotations in batches for memory efficiency
   - Use vectorized projection method to compute AABB extents for entire batches
   - Track ALL fitting solutions (score ≤ 1.0), not just the best
   - Return top-K candidates for refinement

3. **Refinement Phase** (`refinement_phase`)
   - Use Nelder-Mead optimization to locally refine top-K candidates
   - Parameterize rotations as rotation vectors (avoids gimbal lock)
   - Minimize the max ratio of (sorted AABB dims) / (sorted build dims)

4. **Diverse Solution Selection** (`select_diverse`)
   - Select up to 10 maximally diverse fitting orientations using greedy farthest-point sampling
   - Measure diversity using geodesic distance on SO(3): `(R1 * R2.inv()).magnitude()`
   - Refine the diverse candidates and export as separate STL files
   - Provides users with practical printing options (different support/overhang characteristics)

### Key Implementation Details

**Sorted-dimension comparison:** The tool compares `sorted(AABB_extents)` against `sorted(build_volume)`. This correctly solves the axis assignment problem—which model dimension maps to which printer dimension—without combinatorial search. A model fits if and only if the sorted extents fit within the sorted build volume.

**Vectorized AABB computation** (`compute_aabb_extents_batch`): Instead of rotating hull vertices and computing min/max, the tool projects hull vertices onto the 3 axis directions defined by each rotation matrix. This enables batch processing of thousands of rotations using numpy matrix operations.

**Memory management:** Batch size is auto-adjusted based on hull vertex count to stay under ~1GB memory usage during the projection step.

**Rotation representation:** Uses rotation vectors (axis-angle) during optimization to avoid gimbal lock and quaternion/Euler angle singularities.

### File Structure

- `stl_fit.py` - Single-file implementation containing all logic
- `requirements.txt` - Dependencies: numpy, scipy, trimesh
- `.gitignore` - Excludes `.claude/` and generated `*_rotated*.stl` files

### Dependencies

- **numpy**: Array operations and vectorized computations
- **scipy**: Rotation handling (`scipy.spatial.transform.Rotation`) and optimization (`scipy.optimize.minimize`)
- **trimesh**: STL loading, mesh operations, and convex hull computation
