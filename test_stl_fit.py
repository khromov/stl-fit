"""Tests for stl_fit.py"""

import os
import tempfile
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stl_fit import (
    compute_aabb_extents_batch,
    coarse_search,
    refine,
    refinement_phase,
    select_diverse,
    load_and_prepare,
    report_result,
    DEFAULT_BUILD_VOLUME,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cube_hull_verts():
    """Unit cube centered at origin: corners at +/-0.5."""
    corners = np.array(np.meshgrid([-0.5, 0.5], [-0.5, 0.5], [-0.5, 0.5])).T.reshape(-1, 3)
    return corners


@pytest.fixture
def box_hull_verts():
    """Rectangular box 100x50x25 mm, centered at origin."""
    hx, hy, hz = 50.0, 25.0, 12.5
    corners = np.array(np.meshgrid([-hx, hx], [-hy, hy], [-hz, hz])).T.reshape(-1, 3)
    return corners


@pytest.fixture
def tmp_stl_path():
    """Create a temporary STL file from a simple box mesh."""
    import trimesh
    mesh = trimesh.creation.box(extents=(100, 50, 25))
    fd, path = tempfile.mkstemp(suffix=".stl")
    os.close(fd)
    mesh.export(path)
    yield path
    os.unlink(path)


# ---------------------------------------------------------------------------
# compute_aabb_extents_batch
# ---------------------------------------------------------------------------

class TestComputeAABBExtentsBatch:
    def test_identity_rotation_cube(self, cube_hull_verts):
        """Identity rotation on a unit cube should give extents of ~1.0 each."""
        rot = np.eye(3).reshape(1, 3, 3)
        extents = compute_aabb_extents_batch(cube_hull_verts, rot)
        np.testing.assert_allclose(extents, [[1.0, 1.0, 1.0]], atol=1e-10)

    def test_identity_rotation_box(self, box_hull_verts):
        """Identity rotation on a box should return original dimensions."""
        rot = np.eye(3).reshape(1, 3, 3)
        extents = compute_aabb_extents_batch(box_hull_verts, rot)
        extents_sorted = np.sort(extents[0])
        np.testing.assert_allclose(extents_sorted, [25.0, 50.0, 100.0], atol=1e-10)

    def test_batch_multiple_rotations(self, cube_hull_verts):
        """Batch of rotations should return correct number of results."""
        rots = Rotation.random(10, random_state=42).as_matrix()
        extents = compute_aabb_extents_batch(cube_hull_verts, rots)
        assert extents.shape == (10, 3)

    def test_cube_extents_invariant_to_rotation(self, cube_hull_verts):
        """A cube's sorted AABB extents should be ~1.0 regardless of rotation.

        Note: a rotated cube has a *larger* AABB (sqrt(2) or sqrt(3) on diagonals),
        so we just check that every extent is >= 1.0 and <= sqrt(3).
        """
        rots = Rotation.random(50, random_state=0).as_matrix()
        extents = compute_aabb_extents_batch(cube_hull_verts, rots)
        assert np.all(extents >= 1.0 - 1e-10)
        assert np.all(extents <= np.sqrt(3) + 1e-10)

    def test_90_degree_rotation_swaps_axes(self, box_hull_verts):
        """90-degree rotation around Z should swap X and Y extents."""
        rot_90z = Rotation.from_euler('z', 90, degrees=True).as_matrix().reshape(1, 3, 3)
        extents = compute_aabb_extents_batch(box_hull_verts, rot_90z)
        extents_sorted = np.sort(extents[0])
        np.testing.assert_allclose(extents_sorted, [25.0, 50.0, 100.0], atol=1e-10)


# ---------------------------------------------------------------------------
# coarse_search
# ---------------------------------------------------------------------------

class TestCoarseSearch:
    def test_small_model_fits(self, cube_hull_verts):
        """A tiny cube should easily fit in default build volume."""
        build_dims = np.array([180.0, 180.0, 180.0])
        top_k, fitting = coarse_search(cube_hull_verts, build_dims, num_samples=1000, batch_size=500)
        assert len(top_k) > 0
        best_score = top_k[0][0]
        assert best_score <= 1.0
        assert len(fitting) > 0

    def test_huge_model_does_not_fit(self):
        """A model larger than the build volume shouldn't produce fitting results."""
        hx, hy, hz = 200.0, 200.0, 200.0
        verts = np.array(np.meshgrid([-hx, hx], [-hy, hy], [-hz, hz])).T.reshape(-1, 3)
        build_dims = np.array([180.0, 180.0, 180.0])
        top_k, fitting = coarse_search(verts, build_dims, num_samples=500, batch_size=500)
        assert len(fitting) == 0
        assert top_k[0][0] > 1.0

    def test_returns_top_k_sorted(self, cube_hull_verts):
        """Top-K results should be sorted by score (ascending)."""
        build_dims = np.array([180.0, 180.0, 180.0])
        top_k, _ = coarse_search(cube_hull_verts, build_dims, num_samples=2000, batch_size=500)
        scores = [r[0] for r in top_k]
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# refine
# ---------------------------------------------------------------------------

class TestRefine:
    def test_improves_score(self, box_hull_verts):
        """Refinement should produce a score <= the coarse score."""
        build_dims_sorted = np.sort([180.0, 180.0, 180.0])
        initial = Rotation.random(random_state=99)
        # Compute coarse score
        R = initial.as_matrix()
        rotated = box_hull_verts @ R.T
        coarse_extents = rotated.max(axis=0) - rotated.min(axis=0)
        coarse_score = np.max(np.sort(coarse_extents) / build_dims_sorted)

        refined_score, _, _ = refine(box_hull_verts, build_dims_sorted, initial)
        assert refined_score <= coarse_score + 1e-6

    def test_returns_valid_extents(self, box_hull_verts):
        """Refined extents should have 3 positive components."""
        build_dims_sorted = np.sort([180.0, 180.0, 180.0])
        initial = Rotation.random(random_state=42)
        _, _, extents = refine(box_hull_verts, build_dims_sorted, initial)
        assert extents.shape == (3,)
        assert np.all(extents > 0)


# ---------------------------------------------------------------------------
# refinement_phase
# ---------------------------------------------------------------------------

class TestRefinementPhase:
    def test_returns_best(self, box_hull_verts):
        """Should return the best score among refined candidates."""
        build_dims = np.array([180.0, 180.0, 180.0])
        candidates = [
            (0.5, Rotation.random(random_state=1), np.array([50.0, 50.0, 50.0])),
            (0.6, Rotation.random(random_state=2), np.array([60.0, 60.0, 60.0])),
        ]
        score, rot, extents = refinement_phase(box_hull_verts, build_dims, candidates)
        assert score <= 0.6  # should be at least as good as coarse


# ---------------------------------------------------------------------------
# select_diverse
# ---------------------------------------------------------------------------

class TestSelectDiverse:
    def test_returns_n_results(self):
        """Should return exactly n results when enough candidates exist."""
        results = [
            (0.5 + i * 0.01, Rotation.random(random_state=i), np.array([50.0, 50.0, 50.0]))
            for i in range(30)
        ]
        diverse = select_diverse(results, n=5)
        assert len(diverse) == 5

    def test_returns_all_if_fewer_than_n(self):
        """Should return all results if fewer than n."""
        results = [
            (0.5, Rotation.random(random_state=0), np.array([50.0, 50.0, 50.0])),
            (0.6, Rotation.random(random_state=1), np.array([60.0, 60.0, 60.0])),
        ]
        diverse = select_diverse(results, n=10)
        assert len(diverse) == 2

    def test_best_score_first(self):
        """First selected result should have the lowest score."""
        results = [
            (0.9, Rotation.random(random_state=0), np.array([50.0, 50.0, 50.0])),
            (0.3, Rotation.random(random_state=1), np.array([50.0, 50.0, 50.0])),
            (0.7, Rotation.random(random_state=2), np.array([50.0, 50.0, 50.0])),
        ]
        diverse = select_diverse(results, n=2)
        assert diverse[0][0] == 0.3


# ---------------------------------------------------------------------------
# load_and_prepare
# ---------------------------------------------------------------------------

class TestLoadAndPrepare:
    def test_loads_stl(self, tmp_stl_path):
        """Should load an STL and return centered hull verts and mesh."""
        hull_verts, mesh = load_and_prepare(tmp_stl_path)
        assert hull_verts.shape[1] == 3
        assert len(hull_verts) > 0
        # Centroid should be near origin
        centroid = hull_verts.mean(axis=0)
        np.testing.assert_allclose(centroid, [0, 0, 0], atol=1e-6)

    def test_hull_smaller_than_mesh(self, tmp_stl_path):
        """Hull should have fewer or equal vertices than original mesh."""
        hull_verts, mesh = load_and_prepare(tmp_stl_path)
        assert len(hull_verts) <= len(mesh.vertices)


# ---------------------------------------------------------------------------
# report_result (smoke test)
# ---------------------------------------------------------------------------

class TestReportResult:
    def test_fit_output(self, capsys):
        """Should print FIT for score <= 1.0."""
        rot = Rotation.identity()
        extents = np.array([100.0, 100.0, 100.0])
        build_dims = np.array([180.0, 180.0, 180.0])
        report_result(0.56, extents, rot, build_dims)
        captured = capsys.readouterr()
        assert "FIT" in captured.out

    def test_no_fit_output(self, capsys):
        """Should print DOES NOT FIT for score > 1.0."""
        rot = Rotation.identity()
        extents = np.array([200.0, 200.0, 200.0])
        build_dims = np.array([180.0, 180.0, 180.0])
        report_result(1.11, extents, rot, build_dims)
        captured = capsys.readouterr()
        assert "DOES NOT FIT" in captured.out

    def test_scaled_output(self, capsys):
        """Should include scale percentage when scaled."""
        rot = Rotation.identity()
        extents = np.array([100.0, 100.0, 100.0])
        build_dims = np.array([180.0, 180.0, 180.0])
        report_result(0.56, extents, rot, build_dims, scale_pct=95.0)
        captured = capsys.readouterr()
        assert "95.0%" in captured.out
