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

Basic usage (default 180x180x180mm build volume):
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

### Running Tests
```bash
python3 -m pytest test_stl_fit.py -v
```

Run a single test class or method:
```bash
python3 -m pytest test_stl_fit.py::TestComputeAABBExtentsBatch -v
python3 -m pytest test_stl_fit.py::TestRefine::test_improves_score -v
```

Tests generate geometry programmatically (no external STL files needed).

**Note:** The test file imports from `stl_fit` (which re-exported from `stl_fit_lib` in the old single-file layout). After modularization, test imports reference `stl_fit_lib` symbols via `stl_fit`. If adding new library functions, export them from both modules or update test imports.

## Architecture

### File Structure

- `stl_fit.py` — CLI entry point: argument parsing, orchestration loop (coarse search, refinement, scale-down, diverse export)
- `stl_fit_lib.py` — Core library: all algorithm functions and constants
- `test_stl_fit.py` — pytest test suite with fixtures and tests for each core function
- `requirements.txt` — Dependencies: numpy, scipy, trimesh

### Core Algorithm Flow

The tool uses a two-phase optimization approach:

1. **Load & Prepare** (`load_and_prepare`) — Load STL via trimesh, handle multi-body Scene objects, compute convex hull (100-1000x vertex reduction), center at origin.

2. **Coarse Search** (`coarse_search`) — Generate uniform random rotations over SO(3), process in batches, use vectorized projection to compute AABB extents. Tracks ALL fitting solutions (score <= 1.0) and returns top-K candidates.

3. **Refinement Phase** (`refinement_phase`) — Nelder-Mead optimization on top-K candidates, parameterized as rotation vectors to avoid gimbal lock. Minimizes max ratio of sorted(AABB dims) / sorted(build dims).

4. **Iterative Scale-Down** (in `main()`) — If no fit found, shrinks model by 1mm per axis iteratively until it fits, then does an extra shrink for diversity if needed.

5. **Diverse Solution Selection** (`select_diverse`) — Greedy farthest-point sampling using geodesic distance on SO(3): `(R1 * R2.inv()).magnitude()`. Selects up to 10 maximally diverse fitting orientations and exports as separate STL files.

### Key Implementation Details

**Sorted-dimension comparison:** Compares `sorted(AABB_extents)` against `sorted(build_volume)`. This solves the axis assignment problem without combinatorial search. A model fits iff sorted extents fit within sorted build volume.

**Vectorized AABB computation** (`compute_aabb_extents_batch`): Projects hull vertices onto rotation matrix axis directions, enabling batch processing of thousands of rotations via numpy matrix ops. Memory-bounded to ~1GB by auto-adjusting batch size based on hull vertex count.

**Rotation representation:** Uses rotation vectors (axis-angle) during optimization to avoid gimbal lock and singularities.

### Dependencies

- **numpy**: Array operations and vectorized computations
- **scipy**: `scipy.spatial.transform.Rotation` for SO(3) rotations, `scipy.optimize.minimize` for Nelder-Mead refinement
- **trimesh**: STL loading, mesh operations, convex hull computation
