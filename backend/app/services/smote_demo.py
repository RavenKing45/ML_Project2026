"""The SMOTE explainer of the dashboard (``POST /api/v1/smote-demo``; master guide Parts 3 and 10).

The toy dataset is fixed for each ``(scenario, n_legit, n_fraud)``: it is drawn with
``TOY_DATA_SEED``, never with the request's ``seed``, so the frontend can embed a copy. The
default (``two_clusters``, 300 legitimate, 20 fraud) has two separate fraud clusters of 10 points.
It is oversampled with a transparent SMOTE that returns every intermediate step:

1. ``NearestNeighbors(n_neighbors=k + 1)`` over the fraud points; dropping each point itself
   leaves its ``k`` nearest fraud neighbours (``fraud_neighbors``).
2. With ``rng = np.random.RandomState(seed)``: ``picks = rng.randint(0, n_fraud * k, n_new)``,
   then ``gaps = rng.uniform(size=n_new)``. Pick ``p`` means parent ``p // k`` and that parent's
   ``(p % k)``-th neighbour.
3. ``x_new = x_parent + gap * (x_neighbour - x_parent)``: a point on the segment between them.

This is the procedure of imblearn's ``SMOTE`` (``SMOTE._fit_resample`` / ``_make_samples`` in
imbalanced-learn 0.14), which the experiment itself uses: for the same points, ``k``, number of new
points and seed, both give identical synthetic points (``tests/backend/test_smote_demo.py``
checks it). The teaching version only exists to expose each point's parent, neighbour and gap,
which imblearn does not return.

The number of new points is ``n_fraud * oversample_percent / 100`` (the SMOTE paper's N%), or,
without ``oversample_percent``, imblearn's rule for a float ``sampling_strategy``:
``int(n_legit * sampling_strategy - n_fraud)``. A synthetic point is ``in_legit_region`` when
most of its ``LEGIT_REGION_NEIGHBOURS`` nearest original points are legitimate. The
``decision_grid`` holds ``DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=seed)``
predictions on a ``GRID_SIZE`` x ``GRID_SIZE`` grid, trained on the original points and again with
the synthetic points added.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from backend.app.errors import STATUS_CODES, ApiError
from fraudstudy.contracts import (
    DecisionGrid,
    FraudNeighbors,
    Point2D,
    SmoteDemoRequest,
    SmoteDemoResponse,
    SyntheticPoint,
)

TOY_DATA_SEED = 0
DATA_DECIMALS = 3
LEGIT_CENTRE = (0.0, 0.0)
LEGIT_SPREAD = 1.5
GRID_SIZE = 40
GRID_PADDING = 0.5
TREE_MAX_DEPTH = 4
LEGIT_REGION_NEIGHBOURS = 5
PERCENT = 100
SEED_MAX = 2**32 - 1  # NumPy's RandomState (and so imblearn and scikit-learn) accept 0..2**32-1


@dataclass(frozen=True)
class Cluster:
    """A Gaussian cluster of fraud points.

    Attributes:
        centre: ``(x, y)`` mean.
        spread: Standard deviation along each axis.
    """

    centre: tuple[float, float]
    spread: float


SCENARIOS: dict[str, tuple[Cluster, ...]] = {
    "single_cluster": (Cluster((2.8, 2.4), 0.45),),
    "two_clusters": (Cluster((2.8, 2.4), 0.35), Cluster((-2.8, 2.4), 0.35)),
    "overlap": (Cluster((1.0, 0.8), 0.8),),
}
"""Fraud clusters per scenario; legitimate points are ``N(LEGIT_CENTRE, LEGIT_SPREAD^2 I)``."""


@dataclass(frozen=True)
class ToyData:
    """The toy dataset.

    Attributes:
        X: Points, shape ``(n_legit + n_fraud, 2)``: the legitimate points first, then the fraud
            points cluster by cluster.
        y: Labels (1 = fraud).
        cluster: Cluster number of each fraud point, shape ``(n_fraud,)``.
    """

    X: np.ndarray
    y: np.ndarray
    cluster: np.ndarray


@dataclass(frozen=True)
class SmoteSteps:
    """Every step of one SMOTE run. Positions index the fraud points (``X[y == 1]``).

    Attributes:
        neighbours: Each fraud point's ``k`` nearest other fraud points, nearest first, shape
            ``(n_fraud, k)``.
        parents: Parent of each new point, shape ``(n_new,)``.
        partners: The chosen neighbour of each new point, shape ``(n_new,)``.
        gaps: ``lambda`` of each new point, in [0, 1), shape ``(n_new,)``.
        points: The new points, shape ``(n_new, 2)``.
    """

    neighbours: np.ndarray
    parents: np.ndarray
    partners: np.ndarray
    gaps: np.ndarray
    points: np.ndarray


def make_toy_data(scenario: str, n_legit: int, n_fraud: int) -> ToyData:
    """Draw the fixed toy dataset (always with ``TOY_DATA_SEED``).

    Fraud points are split as evenly as possible between the scenario's clusters. Coordinates
    are rounded to ``DATA_DECIMALS`` decimals; SMOTE then runs on the rounded points.

    Args:
        scenario: A key of :data:`SCENARIOS`.
        n_legit: Legitimate points.
        n_fraud: Fraud points.

    Returns:
        The :class:`ToyData`.
    """
    rng = np.random.default_rng(TOY_DATA_SEED)
    legit = rng.normal(LEGIT_CENTRE, LEGIT_SPREAD, size=(n_legit, 2))
    clusters = SCENARIOS[scenario]
    sizes = [len(part) for part in np.array_split(np.arange(n_fraud), len(clusters))]
    fraud = [
        rng.normal(c.centre, c.spread, size=(n, 2)) for c, n in zip(clusters, sizes, strict=True)
    ]
    X = np.round(np.vstack([legit, *fraud]), DATA_DECIMALS)
    y = np.concatenate([np.zeros(n_legit, dtype=int), np.ones(n_fraud, dtype=int)])
    return ToyData(X, y, np.repeat(np.arange(len(clusters)), sizes))


def n_synthetic(request: SmoteDemoRequest) -> int:
    """How many synthetic points to make (rules in the module docstring); never negative."""
    if request.oversample_percent is not None:
        return request.n_fraud * request.oversample_percent // PERCENT
    return max(0, int(request.n_legit * request.sampling_strategy - request.n_fraud))


def smote_steps(X_fraud: np.ndarray, k: int, n_new: int, seed: int) -> SmoteSteps:
    """Run SMOTE on the fraud points, keeping every step (imblearn's procedure, see the module
    docstring).

    Args:
        X_fraud: The fraud points, shape ``(n_fraud, 2)``, ``n_fraud > k``.
        k: Neighbours per point.
        n_new: Synthetic points to make.
        seed: Seed of the ``RandomState`` that picks parents, neighbours and gaps.

    Returns:
        The :class:`SmoteSteps`.
    """
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X_fraud)
    neighbours = nn.kneighbors(X_fraud, return_distance=False)[:, 1:]
    rng = np.random.RandomState(seed)  # what imblearn's check_random_state(seed) returns
    picks = rng.randint(low=0, high=neighbours.size, size=n_new)
    gaps = rng.uniform(size=n_new)
    parents, slots = np.divmod(picks, k)
    partners = neighbours[parents, slots]
    points = X_fraud[parents] + gaps[:, np.newaxis] * (X_fraud[partners] - X_fraud[parents])
    return SmoteSteps(neighbours, parents, partners, gaps, points)


def in_legit_region(X: np.ndarray, y: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Whether most of each point's ``LEGIT_REGION_NEIGHBOURS`` nearest original points are
    legitimate.

    Args:
        X: Original points.
        y: Their labels.
        points: Points to check, shape ``(n, 2)``.

    Returns:
        Booleans, shape ``(n,)``.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool)
    k = min(LEGIT_REGION_NEIGHBOURS, len(X))
    nearest = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(points, return_distance=False)
    return (y[nearest] == 0).sum(axis=1) * 2 > k


def decision_grid(
    X: np.ndarray, y: np.ndarray, X_synthetic: np.ndarray, seed: int
) -> DecisionGrid:
    """Decision-tree predictions on a grid, trained before and after adding the synthetic points.

    The grid spans the original points plus ``GRID_PADDING`` on every side (synthetic points lie
    between fraud points, so inside it). ``before[i][j]`` is the label at ``(x[j], y[i])``.

    Args:
        X: Original points.
        y: Their labels.
        X_synthetic: Synthetic fraud points.
        seed: ``random_state`` of both trees.

    Returns:
        The :class:`DecisionGrid`.
    """
    low, high = X.min(axis=0) - GRID_PADDING, X.max(axis=0) + GRID_PADDING
    xs, ys = np.linspace(low[0], high[0], GRID_SIZE), np.linspace(low[1], high[1], GRID_SIZE)
    grid = np.column_stack([axis.ravel() for axis in np.meshgrid(xs, ys)])

    def labels(points: np.ndarray, targets: np.ndarray) -> list[list[int]]:
        tree = DecisionTreeClassifier(max_depth=TREE_MAX_DEPTH, random_state=seed)
        return tree.fit(points, targets).predict(grid).reshape(GRID_SIZE, GRID_SIZE).tolist()

    after_y = np.concatenate([y, np.ones(len(X_synthetic), dtype=int)])
    return DecisionGrid(
        model=f"DecisionTreeClassifier(max_depth={TREE_MAX_DEPTH})",
        x=xs.tolist(),
        y=ys.tolist(),
        before=labels(X, y),
        after=labels(np.vstack([X, X_synthetic]), after_y),
    )


def demo_warnings(
    request: SmoteDemoRequest, data: ToyData, steps: SmoteSteps, n_in_legit: int
) -> list[str]:
    """Teaching notes for the response.

    Args:
        request: The validated request.
        data: The toy dataset.
        steps: The SMOTE run.
        n_in_legit: Synthetic points in the legitimate region.

    Returns:
        In order: ``sampling_strategy`` ignored (both given), nothing generated, neighbours that
        join different fraud clusters, synthetic points among legitimate ones.
    """
    warnings: list[str] = []
    if request.oversample_percent is not None and "sampling_strategy" in request.model_fields_set:
        warnings.append("sampling_strategy is ignored because oversample_percent is given.")
    if len(steps.points) == 0:
        current = request.n_fraud / request.n_legit
        warnings.append(
            f"sampling_strategy {request.sampling_strategy:g} is not above the current "
            f"fraud:legit ratio ({current:.3f}), so no synthetic points were made."
        )
    if (data.cluster[steps.neighbours] != data.cluster[:, np.newaxis]).any():
        warnings.append(
            f"k = {request.k_neighbors} links points of different fraud clusters: large k can "
            "connect distant fraud clusters, creating synthetic points in legitimate territory."
        )
    if n_in_legit:
        warnings.append(
            f"{n_in_legit} of {len(steps.points)} synthetic points landed among legitimate points "
            f"(most of their {LEGIT_REGION_NEIGHBOURS} nearest original points are legitimate)."
        )
    return warnings


def run_smote_demo(request: SmoteDemoRequest) -> SmoteDemoResponse:
    """Build the toy data, run SMOTE step by step and train the before / after trees.

    Indices (``parent``, ``neighbor``, ``fraud_neighbors``) refer to ``original``, in which the
    fraud points follow the ``n_legit`` legitimate ones.

    Args:
        request: The validated request.

    Returns:
        The ``SmoteDemoResponse``.

    Raises:
        ApiError: 422 if ``seed`` is outside 0..2**32 - 1.
    """
    if not 0 <= request.seed <= SEED_MAX:
        raise ApiError(
            422,
            STATUS_CODES[422],
            f"seed must be between 0 and {SEED_MAX} (NumPy's RandomState range).",
            [{"loc": ["body", "seed"], "msg": "Out of range", "type": "value_error"}],
        )
    data = make_toy_data(request.scenario, request.n_legit, request.n_fraud)
    offset = request.n_legit
    steps = smote_steps(data.X[offset:], request.k_neighbors, n_synthetic(request), request.seed)
    legit_region = in_legit_region(data.X, data.y, steps.points)
    n_in_legit = int(legit_region.sum())
    return SmoteDemoResponse(
        request=request,
        original=[
            Point2D(x=float(x), y=float(y), label=int(label))
            for (x, y), label in zip(data.X, data.y, strict=True)
        ],
        synthetic=[
            SyntheticPoint(
                x=float(point[0]),
                y=float(point[1]),
                parent=offset + int(parent),
                neighbor=offset + int(partner),
                gap=float(gap),
                in_legit_region=bool(in_legit),
            )
            for point, parent, partner, gap, in_legit in zip(
                steps.points, steps.parents, steps.partners, steps.gaps, legit_region, strict=True
            )
        ],
        n_synthetic_in_legit_region=n_in_legit,
        warnings=demo_warnings(request, data, steps, n_in_legit),
        fraud_neighbors=[
            FraudNeighbors(point=offset + i, neighbors=(offset + row).tolist())
            for i, row in enumerate(steps.neighbours)
        ],
        decision_grid=decision_grid(data.X, data.y, steps.points, request.seed),
    )
