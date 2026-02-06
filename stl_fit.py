#!/usr/bin/env python3
"""Check if an STL model fits in a build volume by searching over rotations."""

import argparse
import sys
import time
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize

try:
    import trimesh
except ImportError:
    print("Error: trimesh is not installed. Run: pip install numpy scipy trimesh")
    sys.exit(2)


# Constants
DEFAULT_BUILD_VOLUME = (180.0, 180.0, 180.0)
DEFAULT_NUM_SAMPLES = 500_000
DEFAULT_BATCH_SIZE = 5_000
DEFAULT_NUM_REFINE_STARTS = 5


def load_and_prepare(stl_path):
    """Load STL, compute convex hull, and return centered hull vertices and mesh."""
    try:
        mesh = trimesh.load(stl_path)
    except Exception as e:
        print(f"Error loading STL file: {e}")
        sys.exit(2)

    # Handle Scene vs Mesh
    if isinstance(mesh, trimesh.Scene):
        print("  Multi-body STL detected, concatenating...")
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    print(f"Loading STL: {stl_path}")
    print(f"  Triangles:       {len(mesh.faces):,}")
    print(f"  Vertices:        {len(mesh.vertices):,}")

    # Compute original AABB
    original_extents = mesh.bounds[1] - mesh.bounds[0]
    print(f"  Original AABB:   {original_extents[0]:.1f} x {original_extents[1]:.1f} x {original_extents[2]:.1f} mm")

    # Compute convex hull
    hull = mesh.convex_hull
    hull_verts = hull.vertices.copy()
    print(f"  Convex hull:     {len(hull_verts):,} vertices")

    # Center hull vertices (and mesh) at origin
    centroid = hull_verts.mean(axis=0)
    hull_verts -= centroid
    mesh.vertices -= centroid

    return hull_verts, mesh


def compute_aabb_extents_batch(hull_verts, rot_matrices):
    """Compute AABB extents for a batch of rotations using the projection method.

    Args:
        hull_verts: (N, 3) convex hull vertices
        rot_matrices: (B, 3, 3) rotation matrices

    Returns:
        extents: (B, 3) AABB extents for each rotation
    """
    B = rot_matrices.shape[0]

    # Flatten rotation matrices to get all axis directions
    directions = rot_matrices.reshape(-1, 3)  # (B*3, 3)

    # Project all hull vertices onto all directions
    projections = hull_verts @ directions.T  # (N, B*3)

    # Compute min/max along each direction and reshape
    proj_max = projections.max(axis=0).reshape(B, 3)
    proj_min = projections.min(axis=0).reshape(B, 3)

    extents = proj_max - proj_min
    return extents


def coarse_search(hull_verts, build_dims, num_samples, batch_size):
    """Perform coarse SO(3) search over random rotations.

    Returns:
        top_results: list of (score, rotation, extents) sorted by score
    """
    print(f"\nCoarse search: {num_samples:,} random rotations (batch size {batch_size:,})")

    # Generate all rotations at once
    rotations = Rotation.random(num_samples)
    rot_matrices = rotations.as_matrix()

    build_dims_sorted = np.sort(build_dims)

    # Track top results (score, rotation_idx, extents)
    top_results = []
    best_score = float('inf')

    num_batches = (num_samples + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_samples)
        batch_rot_mats = rot_matrices[start_idx:end_idx]

        # Compute AABB extents for this batch
        extents = compute_aabb_extents_batch(hull_verts, batch_rot_mats)

        # Sort extents and compare
        extents_sorted = np.sort(extents, axis=1)
        ratios = extents_sorted / build_dims_sorted
        scores = ratios.max(axis=1)

        # Find best in this batch
        batch_best_idx = scores.argmin()
        batch_best_score = scores[batch_best_idx]

        if batch_best_score < best_score:
            best_score = batch_best_score
            global_idx = start_idx + batch_best_idx
            top_results.append((
                batch_best_score,
                rotations[global_idx],
                extents[batch_best_idx]
            ))

        # Progress indicator
        if (batch_idx + 1) % max(1, num_batches // 10) == 0 or batch_idx == num_batches - 1:
            elapsed = time.time() - start_time
            progress = (batch_idx + 1) / num_batches * 100
            print(f"  [{int(progress):3d}%] {batch_idx + 1}/{num_batches} batches ({elapsed:.1f}s) | Best score: {best_score:.4f}")

    # Sort and keep top-K
    top_results.sort(key=lambda x: x[0])
    top_k = top_results[:DEFAULT_NUM_REFINE_STARTS]

    fits = best_score <= 1.0
    print(f"  Best score: {best_score:.4f} {'(fit found!)' if fits else '(no fit in coarse search)'}")

    return top_k


def refine(hull_verts, build_dims_sorted, initial_rotation):
    """Refine a rotation using Nelder-Mead optimization."""

    def objective(rotvec):
        R = Rotation.from_rotvec(rotvec).as_matrix()
        rotated = hull_verts @ R.T
        extents = rotated.max(axis=0) - rotated.min(axis=0)
        sorted_ext = np.sort(extents)
        return np.max(sorted_ext / build_dims_sorted)

    initial_rotvec = initial_rotation.as_rotvec()

    result = minimize(
        objective,
        initial_rotvec,
        method='Nelder-Mead',
        options={'maxiter': 10000, 'xatol': 1e-6, 'fatol': 1e-8}
    )

    # Compute final extents
    final_rotation = Rotation.from_rotvec(result.x)
    R = final_rotation.as_matrix()
    rotated = hull_verts @ R.T
    final_extents = rotated.max(axis=0) - rotated.min(axis=0)

    return result.fun, final_rotation, final_extents


def refinement_phase(hull_verts, build_dims, top_k_results):
    """Refine top-K results and return the best."""
    print(f"\nRefining from top {len(top_k_results)} candidates...")

    build_dims_sorted = np.sort(build_dims)

    refined_results = []
    for i, (coarse_score, rotation, _) in enumerate(top_k_results):
        score, refined_rot, extents = refine(hull_verts, build_dims_sorted, rotation)
        refined_results.append((score, refined_rot, extents))
        print(f"  Start {i+1}: coarse {coarse_score:.4f} -> refined {score:.4f}")

    # Return best
    refined_results.sort(key=lambda x: x[0])
    best_score, best_rotation, best_extents = refined_results[0]
    print(f"  Best refined score: {best_score:.4f}")

    return best_score, best_rotation, best_extents


def report_result(score, extents, rotation, build_dims):
    """Print final formatted result."""
    build_dims_sorted = np.sort(build_dims)
    extents_sorted = np.sort(extents)

    fits = score <= 1.0

    print("\n" + "=" * 60)
    if fits:
        print("  RESULT: FIT")
    else:
        print("  RESULT: DOES NOT FIT")
    print("=" * 60)

    print(f"  Bounding box:  {extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm")
    print(f"                 (sorted: {extents_sorted[0]:.1f}, {extents_sorted[1]:.1f}, {extents_sorted[2]:.1f})")
    print(f"  Build volume:  {build_dims[0]:.1f} x {build_dims[1]:.1f} x {build_dims[2]:.1f} mm")
    print(f"                 (sorted: {build_dims_sorted[0]:.1f}, {build_dims_sorted[1]:.1f}, {build_dims_sorted[2]:.1f})")

    if fits:
        margins = build_dims_sorted - extents_sorted
        print(f"  Margins:       +{margins[0]:.1f}, +{margins[1]:.1f}, +{margins[2]:.1f} mm")
    else:
        excess = extents_sorted - build_dims_sorted
        print(f"  Excess:        {excess[0]:.1f}, {excess[1]:.1f}, {excess[2]:.1f} mm")
        print(f"  Score:         {score:.4f} (needs {(score - 1.0) * 100:.2f}% reduction to fit)")

    # Print rotation
    euler = rotation.as_euler('zyx', degrees=True)
    rotvec = rotation.as_rotvec()
    print(f"\n  Rotation (Euler ZYX): yaw={euler[0]:.1f}°, pitch={euler[1]:.1f}°, roll={euler[2]:.1f}°")
    print(f"  Rotation (rotvec):    [{rotvec[0]:.3f}, {rotvec[1]:.3f}, {rotvec[2]:.3f}]")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Check if an STL model fits in a build volume by searching over rotations."
    )
    parser.add_argument("--stl", required=True, help="Path to STL file")
    parser.add_argument(
        "--build-volume",
        type=float,
        nargs=3,
        default=DEFAULT_BUILD_VOLUME,
        metavar=("X", "Y", "Z"),
        help="Build volume dimensions in mm (default: 180 180 180)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of random rotations for coarse search (default: {DEFAULT_NUM_SAMPLES:,})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Rotations per batch (default: {DEFAULT_BATCH_SIZE:,})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for rotated STL (e.g., rotated.stl)"
    )

    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    build_dims = np.array(args.build_volume)
    print(f"Build volume:      {build_dims[0]:.1f} x {build_dims[1]:.1f} x {build_dims[2]:.1f} mm")

    # Phase 1: Load and prepare
    hull_verts, mesh = load_and_prepare(args.stl)

    # Auto-adjust batch size if needed for memory
    max_memory_bytes = 1e9  # 1 GB
    n_hull = len(hull_verts)
    max_batch = int(max_memory_bytes / (n_hull * 3 * 8))
    if args.batch_size > max_batch:
        print(f"  Warning: Reducing batch size to {max_batch:,} for memory constraints")
        args.batch_size = max_batch

    # Phase 2: Coarse search
    top_k_results = coarse_search(hull_verts, build_dims, args.samples, args.batch_size)

    # Phase 3: Refinement
    best_score, best_rotation, best_extents = refinement_phase(hull_verts, build_dims, top_k_results)

    # Report
    report_result(best_score, best_extents, best_rotation, build_dims)

    # Export rotated mesh if requested
    if args.output:
        print(f"\nExporting rotated mesh to: {args.output}")
        rotated_mesh = mesh.copy()
        rotation_matrix = best_rotation.as_matrix()
        rotated_mesh.vertices = rotated_mesh.vertices @ rotation_matrix.T
        rotated_mesh.export(args.output)
        print(f"  Saved successfully!")

    # Exit with appropriate code
    sys.exit(0 if best_score <= 1.0 else 1)


if __name__ == "__main__":
    main()
