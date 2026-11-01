"""Topological polygon validation, independently of image or COCO parsing."""

import math

import pytest


def _validate(points):
    from tiangong.source_geometry import validate_simple_polygon

    return validate_simple_polygon(points)


@pytest.mark.parametrize("points", [
    [(0, 0), (4, 4), (0, 4), (3, 0)],  # Proper crossing; nonzero shoelace area.
    [(0, 0), (4, 0), (4, 4), (2, 0), (0, 4)],  # Non-adjacent vertex on edge.
    [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0), (-1, 2)],  # Repeated internal vertex.
    [(0, 0), (4, 0), (4, 0), (0, 4)],  # Zero-length edge.
    [(0, 0), (4, 0), (2, 0), (2, 4), (0, 4)],  # Collinear backtracking.
    [(0, 0), (2, 0), (4, 0)],  # All vertices collinear.
    [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0), (0, 0)],  # Double closure.
])
def test_rejects_intersecting_touching_repeated_and_backtracking_edges(points):
    with pytest.raises(ValueError):
        _validate(points)


@pytest.mark.parametrize("points", [
    [(0, 0), (4, 0), (4, 4), (0, 4)],
    [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
    [(0, 0), (2, 0), (4, 0), (4, 4), (0, 4)],
    [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)],  # Concave is valid.
    [(0, 0), (1, 0), (1, 1e-100), (0, 1e-100)],  # No global epsilon.
    [(0, 0), (1e-200, 0), (1e-200, 1e-200), (0, 1e-200)],  # Product underflow.
    [(10**20, 10**20), (10**20 + 1, 10**20), (10**20 + 1, 10**20 + 1), (10**20, 10**20 + 1)],
])
def test_accepts_simple_polygons_without_mutating_vertices(points):
    original = list(points)
    assert _validate(points) is None
    assert points == original
    assert _validate(list(reversed(points))) is None


@pytest.mark.parametrize("points", [
    [], [(0, 0), (1, 1)], [(0, 0), (1, 1), (math.nan, 2)],
    [(0, 0), (1, 1), (math.inf, 2)], [(0, 0), (1, 1), (True, 2)],
    [(0, 0), (1, 1), ("1", 2)], [(0, 0), (1, 1), (2,)],
])
def test_rejects_missing_malformed_and_nonfinite_vertices(points):
    with pytest.raises(ValueError):
        _validate(points)


def test_vertex_budget_accepts_limit_and_rejects_above_before_pair_scan():
    points = [(math.cos(index * 2 * math.pi / 1024), math.sin(index * 2 * math.pi / 1024)) for index in range(1024)]
    assert _validate(points) is None
    assert _validate(points + points[:1]) is None
    over_limit = [(math.cos(index * 2 * math.pi / 1025), math.sin(index * 2 * math.pi / 1025)) for index in range(1025)]
    with pytest.raises(ValueError, match="1024"):
        _validate(over_limit)
