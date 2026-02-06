#!/usr/bin/env python3
"""Check if an STL model fits in a build volume by searching over rotations."""

import argparse
import os
import sys
import numpy as np

from stl_fit_lib import (
    DEFAULT_NUM_SAMPLES,
    DEFAULT_BATCH_SIZE,
    load_and_prepare,
    coarse_search,
    refinement_phase,
    refine,
    select_diverse,
    report_result,
)


def main():
    parser = argparse.ArgumentParser(
        description="Check if an STL model fits in a build volume by searching over rotations."
    )
    parser.add_argument("--stl", required=True, help="Path to STL file")
    parser.add_argument(
        "--build-volume",
        type=float,
        default=180.0,
        metavar="SIZE",
        help="Build volume cube size in mm (default: 180)"
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

    # Set default output path based on input filename if not provided
    if args.output is None:
        input_stem, input_ext = os.path.splitext(args.stl)
        args.output = f"{input_stem}_rotated{input_ext or '.stl'}"

    # Build volume is a cube
    build_dims = np.array([args.build_volume, args.build_volume, args.build_volume])
    print(f"Build volume:      {build_dims[0]:.1f} x {build_dims[1]:.1f} x {build_dims[2]:.1f} mm")

    # Phase 1: Load and prepare
    hull_verts, mesh = load_and_prepare(args.stl)

    # Auto-adjust batch size if needed for memory
    max_memory_bytes = 1e9  # 1 GB
    n_hull = len(hull_verts)
    auto_batch = int(max_memory_bytes / (n_hull * 3 * 8))
    if args.batch_size == DEFAULT_BATCH_SIZE:
        # Auto-scale: use maximum batch that fits in memory
        args.batch_size = auto_batch
    elif args.batch_size > auto_batch:
        print(f"  Warning: Reducing batch size to {auto_batch:,} for memory constraints")
        args.batch_size = auto_batch

    # Phase 2: Coarse search
    top_k_results, fitting_results = coarse_search(hull_verts, build_dims, args.samples, args.batch_size)

    # Phase 3: Refinement
    best_score, best_rotation, best_extents = refinement_phase(hull_verts, build_dims, top_k_results)

    # Phase 3.5: Iterative scale-down if needed
    scale_pct = 100.0

    while best_score > 1.0:
        # Compute how much to shrink: reduce the bounding box by 1mm on each axis
        # Current best extents tell us the model size; we want it 1mm smaller per axis
        current_max_extent = np.sort(best_extents)[-1]
        shrink = (current_max_extent - 1.0) / current_max_extent
        hull_verts = hull_verts * shrink
        mesh.vertices = mesh.vertices * shrink
        scale_pct *= shrink

        print(f"\n--- Shrinking by 1mm per axis (now {scale_pct:.1f}% of original) ---")

        # Re-run full search
        top_k_results, fitting_results = coarse_search(hull_verts, build_dims, args.samples, args.batch_size)
        best_score, best_rotation, best_extents = refinement_phase(hull_verts, build_dims, top_k_results)

    # If we had to scale and don't have enough diverse candidates, shrink once more
    if scale_pct < 100.0 and len(fitting_results) < 10:
        current_max_extent = np.sort(best_extents)[-1]
        shrink = (current_max_extent - 1.0) / current_max_extent
        hull_verts = hull_verts * shrink
        mesh.vertices = mesh.vertices * shrink
        scale_pct *= shrink

        print(f"\n--- Extra shrink for diversity (now {scale_pct:.1f}% of original) ---")

        top_k_results, fitting_results = coarse_search(hull_verts, build_dims, args.samples, args.batch_size)
        best_score, best_rotation, best_extents = refinement_phase(hull_verts, build_dims, top_k_results)

    # Report best result
    report_result(best_score, best_extents, best_rotation, build_dims, scale_pct=scale_pct)

    # Export rotated meshes
    build_dims_sorted = np.sort(build_dims)

    # If model fits, find diverse solutions
    if best_score <= 1.0 and len(fitting_results) > 0:
        # Select up to 10 diverse solutions from coarse results
        diverse_coarse = select_diverse(fitting_results, n=min(10, len(fitting_results)))

        # Refine only the selected 10
        print(f"\nRefining {len(diverse_coarse)} diverse candidates...")
        diverse_solutions = []
        for i, (coarse_score, rotation, _) in enumerate(diverse_coarse, start=1):
            score, refined_rot, extents = refine(hull_verts, build_dims_sorted, rotation)
            # Only keep if still fits after refinement
            if score <= 1.0:
                diverse_solutions.append((score, refined_rot, extents))
                print(f"  Solution {i}: coarse {coarse_score:.4f} -> refined {score:.4f}")
            else:
                print(f"  Solution {i}: coarse {coarse_score:.4f} -> refined {score:.4f} (no longer fits, skipped)")

        if len(diverse_solutions) > 0:
            # Extract stem and extension from output path
            output_base = os.path.splitext(args.output)[0]
            output_ext = os.path.splitext(args.output)[1] or '.stl'

            # Add scale tag if model was scaled
            if scale_pct < 100.0:
                output_base = f"{output_base}_{scale_pct:.0f}pct"

            print(f"\n{'=' * 60}")
            print(f"  EXPORTING {len(diverse_solutions)} DIVERSE ORIENTATIONS")
            print(f"{'=' * 60}")

            # Export each diverse solution
            for i, (score, rotation, extents) in enumerate(diverse_solutions, start=1):
                output_path = f"{output_base}_{i}{output_ext}"

                # Compute metrics
                extents_sorted = np.sort(extents)
                margins = build_dims_sorted - extents_sorted
                euler = rotation.as_euler('zyx', degrees=True)

                # Compute distance from first solution
                if i == 1:
                    geodesic_dist = 0.0
                else:
                    geodesic_dist = (rotation * diverse_solutions[0][1].inv()).magnitude()

                # Print summary
                print(f"\nSolution {i}:")
                print(f"  File:      {output_path}")
                print(f"  AABB:      {extents_sorted[0]:.1f} x {extents_sorted[1]:.1f} x {extents_sorted[2]:.1f} mm")
                print(f"  Margins:   +{margins[0]:.1f}, +{margins[1]:.1f}, +{margins[2]:.1f} mm")
                print(f"  Euler:     yaw={euler[0]:.1f}°, pitch={euler[1]:.1f}°, roll={euler[2]:.1f}°")
                print(f"  Distance:  {geodesic_dist:.3f} rad ({np.degrees(geodesic_dist):.1f}° from solution 1)")

                # Export
                rotated_mesh = mesh.copy()
                rotation_matrix = rotation.as_matrix()
                rotated_mesh.vertices = rotated_mesh.vertices @ rotation_matrix.T
                rotated_mesh.export(output_path)

            print(f"\n{'=' * 60}")
            print(f"  Saved {len(diverse_solutions)} rotated STL files successfully!")
            print(f"{'=' * 60}")
        else:
            print("\nWarning: No fitting solutions remained after refinement.")
    else:
        # Model doesn't fit - export single best attempt anyway
        output_path = args.output
        if scale_pct < 100.0:
            output_stem, output_ext = os.path.splitext(args.output)
            output_path = f"{output_stem}_{scale_pct:.0f}pct{output_ext or '.stl'}"

        print(f"\nModel does not fit, but exporting best attempt to: {output_path}")
        rotated_mesh = mesh.copy()
        rotation_matrix = best_rotation.as_matrix()
        rotated_mesh.vertices = rotated_mesh.vertices @ rotation_matrix.T
        rotated_mesh.export(output_path)
        print(f"  Saved!")

    # Exit with appropriate code
    sys.exit(0 if best_score <= 1.0 else 1)


if __name__ == "__main__":
    main()
