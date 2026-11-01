"""Shared polygon topology checks; validation never repairs source geometry."""

from fractions import Fraction
import math
import sys


MAX_POLYGON_VERTICES = 1024
_ORIENTATION_ERROR_BOUND = (3 + 16 * (sys.float_info.epsilon / 2)) * (sys.float_info.epsilon / 2)


def _sign(value) -> int:
    return (value > 0) - (value < 0)


def _exact_orientation(a, b, c) -> int:
    """Exact sign for the actual input floats/integers, without an epsilon."""
    ax, ay = map(Fraction, a)
    bx, by = map(Fraction, b)
    cx, cy = map(Fraction, c)
    return _sign((ax - cx) * (by - cy) - (ay - cy) * (bx - cx))


def _orientation(a, b, c) -> int:
    """Use a bounded floating determinant, falling back near cancellation.

    A fixed geometric tolerance would classify a thin valid polygon as a line.
    Exact arithmetic also handles underflow and large mixed integer inputs.
    """
    values = (*a, *b, *c)
    if all(type(value) is int for value in values):
        return _sign((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) * (b[0] - c[0]))
    if any(type(value) is int and abs(value) > 2**53 for value in values):
        return _exact_orientation(a, b, c)
    ax, ay, bx, by, cx, cy = map(float, values)
    left = (ax - cx) * (by - cy)
    right = (ay - cy) * (bx - cx)
    determinant = left - right
    magnitude = abs(left) + abs(right)
    if (
        math.isfinite(magnitude) and math.isfinite(determinant)
        and (left == 0 or abs(left) >= sys.float_info.min)
        and (right == 0 or abs(right) >= sys.float_info.min)
        and abs(determinant) >= sys.float_info.min
        and abs(determinant) > _ORIENTATION_ERROR_BOUND * magnitude
    ):
        return _sign(determinant)
    return _exact_orientation(a, b, c)


def _intersects(a, b, c, d) -> bool:
    # Reject disjoint bounding boxes before the more expensive orientation
    # predicates. Inclusive comparisons retain endpoint and collinear touches.
    if (
        max(a[0], b[0]) < min(c[0], d[0]) or max(c[0], d[0]) < min(a[0], b[0])
        or max(a[1], b[1]) < min(c[1], d[1]) or max(c[1], d[1]) < min(a[1], b[1])
    ):
        return False
    return (
        _orientation(a, b, c) * _orientation(a, b, d) <= 0
        and _orientation(c, d, a) * _orientation(c, d, b) <= 0
    )


def validate_simple_polygon(points: list[tuple[int | float, int | float]]) -> None:
    """Reject self-intersections, touches, repeated vertices and zero edges.

    A single closing copy of the first vertex is allowed, as are consecutive
    collinear vertices progressing along an edge. Adjacent backtracking and all
    non-adjacent contacts are rejected. Input is preserved exactly. The bounded
    quadratic scan accepts at most 1024 vertices, excluding an optional closure.
    """
    if not isinstance(points, list) or not 3 <= len(points) <= MAX_POLYGON_VERTICES + 1:
        raise ValueError("polygon requires 3 to 1024 vertices, plus optional closure")
    vertices = []
    for point in points:
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise ValueError("polygon vertex must contain exactly two finite coordinates")
        for value in point:
            if type(value) not in (int, float):
                raise ValueError("polygon coordinates must be finite numbers")
            try:
                finite = math.isfinite(value)
            except OverflowError as exc:
                raise ValueError("polygon coordinates exceed finite numeric range") from exc
            if not finite:
                raise ValueError("polygon coordinates must be finite numbers")
        vertices.append(tuple(point))
    if vertices[-1] == vertices[0]:
        vertices.pop()
    count = len(vertices)
    if not 3 <= count <= MAX_POLYGON_VERTICES:
        raise ValueError("polygon requires 3 to 1024 vertices, excluding optional closure")
    if len(set(vertices)) != count:
        raise ValueError("polygon contains repeated vertices or zero-length edges")
    for index, current in enumerate(vertices):
        previous, following = vertices[index - 1], vertices[(index + 1) % count]
        if _orientation(previous, current, following) == 0 and not (
            min(previous[0], following[0]) <= current[0] <= max(previous[0], following[0])
            and min(previous[1], following[1]) <= current[1] <= max(previous[1], following[1])
        ):
            raise ValueError("polygon adjacent edges overlap or backtrack")
    for index, start in enumerate(vertices):
        end = vertices[(index + 1) % count]
        for other in range(index + 2, count):
            if index == 0 and other == count - 1:
                continue  # The closing edge is adjacent to the first edge.
            if _intersects(start, end, vertices[other], vertices[(other + 1) % count]):
                raise ValueError("polygon non-adjacent edges intersect or touch")
