#!/usr/bin/env python3
"""Flask API for stl-fit: optimize STL orientation for a build volume."""

import os
import tempfile

import numpy as np
from flask import Flask, request, send_file, jsonify

from stl_fit_lib import (
    DEFAULT_NUM_SAMPLES,
    load_and_prepare,
    coarse_search,
    refinement_phase,
    refine,
)

app = Flask(__name__)


@app.route("/optimize", methods=["POST"])
def optimize():
    # --- Validate input ---
    if "file" not in request.files:
        return jsonify({"error": "Missing 'file' field"}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"error": "No file selected"}), 400

    build_volume = request.form.get("build_volume", 180.0, type=float)
    samples = request.form.get("samples", DEFAULT_NUM_SAMPLES, type=int)
    seed = request.form.get("seed", None, type=int)

    if build_volume <= 0:
        return jsonify({"error": "build_volume must be positive"}), 400
    if samples <= 0:
        return jsonify({"error": "samples must be positive"}), 400

    if seed is not None:
        np.random.seed(seed)

    # --- Save upload to temp file ---
    tmp_input = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    tmp_output = tempfile.NamedTemporaryFile(suffix=".stl", delete=False)
    tmp_input_path = tmp_input.name
    tmp_output_path = tmp_output.name
    tmp_input.close()
    tmp_output.close()

    try:
        uploaded.save(tmp_input_path)

        # --- Load & prepare ---
        try:
            hull_verts, mesh = load_and_prepare(tmp_input_path)
        except SystemExit:
            return jsonify({"error": "Failed to load STL file"}), 400

        build_dims = np.array([build_volume, build_volume, build_volume])

        # --- Auto-adjust batch size (same memory logic as CLI) ---
        max_memory_bytes = 1e9  # 1 GB
        n_hull = len(hull_verts)
        batch_size = int(max_memory_bytes / (n_hull * 3 * 8))

        # --- Coarse search ---
        top_k_results, fitting_results = coarse_search(
            hull_verts, build_dims, samples, batch_size
        )

        # --- Refinement ---
        best_score, best_rotation, best_extents = refinement_phase(
            hull_verts, build_dims, top_k_results
        )

        # --- Iterative scale-down if needed ---
        scale_pct = 100.0

        while best_score > 1.0:
            current_max_extent = np.sort(best_extents)[-1]
            shrink = (current_max_extent - 1.0) / current_max_extent
            hull_verts = hull_verts * shrink
            mesh.vertices = mesh.vertices * shrink
            scale_pct *= shrink

            top_k_results, fitting_results = coarse_search(
                hull_verts, build_dims, samples, batch_size
            )
            best_score, best_rotation, best_extents = refinement_phase(
                hull_verts, build_dims, top_k_results
            )

        # --- Apply best rotation and export ---
        rotated_mesh = mesh.copy()
        rotated_mesh.vertices = rotated_mesh.vertices @ best_rotation.as_matrix().T
        rotated_mesh.export(tmp_output_path)

        # --- Build response headers ---
        extents_sorted = np.sort(best_extents)
        response = send_file(
            tmp_output_path,
            mimetype="application/octet-stream",
            as_attachment=True,
            download_name="rotated.stl",
        )
        response.headers["X-STL-Fit-Score"] = f"{best_score:.4f}"
        response.headers["X-STL-Fit-Scale-Percent"] = f"{scale_pct:.1f}"
        response.headers["X-STL-Fit-Extents"] = (
            f"{extents_sorted[0]:.1f}x{extents_sorted[1]:.1f}x{extents_sorted[2]:.1f}"
        )

        # Clean up input temp (output cleaned up by send_file / after_request)
        os.unlink(tmp_input_path)

        @response.call_on_close
        def _cleanup():
            try:
                os.unlink(tmp_output_path)
            except OSError:
                pass

        return response

    except Exception as e:
        # Clean up on error
        for path in (tmp_input_path, tmp_output_path):
            try:
                os.unlink(path)
            except OSError:
                pass
        return jsonify({"error": f"Processing failed: {e}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=8000)
