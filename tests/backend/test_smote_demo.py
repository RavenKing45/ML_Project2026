"""Tests for ``POST /api/v1/smote-demo`` and its teaching SMOTE (Prompt 13).

The endpoint needs no artifact, so most tests use the app on the example artifacts.
"""

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from sklearn.tree import DecisionTreeClassifier

from backend.app.main import API_PREFIX, create_app
from backend.app.ratelimit import SMOTE_DEMO_LIMIT, limiter
from backend.app.services import smote_demo as sd
from backend.app.settings import Settings
from fraudstudy.config import PROJECT_ROOT
from fraudstudy.contracts import ErrorResponse, SmoteDemoRequest, SmoteDemoResponse

pytestmark = pytest.mark.usefixtures("rate_limits_off")

DEMO = f"{API_PREFIX}/smote-demo"
EXAMPLES = PROJECT_ROOT / "docs" / "contracts" / "examples"
N_LEGIT, N_FRAUD, CLUSTER_SIZE = 300, 20, 10  # the default request's toy dataset
TOL = 1e-9


def settings(**overrides: Any) -> Settings:
    """Example artifacts, no ``.env``, rate limits off unless overridden."""
    values = {"use_example_artifacts": True, "examples_dir": EXAMPLES, "rate_limit_enabled": False}
    return Settings(_env_file=None, **(values | overrides))


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    with TestClient(create_app(settings())) as test_client:
        yield test_client


def demo(client: TestClient, **body: Any) -> SmoteDemoResponse:
    """POST ``body`` and return the validated 200 response."""
    response = client.post(DEMO, json=body)
    assert response.status_code == 200, response.text
    return SmoteDemoResponse.model_validate(response.json())


def coordinates(result: SmoteDemoResponse) -> np.ndarray:
    """The original points as an ``(n, 2)`` array."""
    return np.array([[p.x, p.y] for p in result.original])


def test_default_request_uses_the_fixed_two_cluster_dataset(client: TestClient) -> None:
    empty_body = SmoteDemoResponse.model_validate(client.post(DEMO).json())
    result = demo(client)
    assert result == empty_body
    labels = [p.label for p in result.original]
    assert labels == [0] * N_LEGIT + [1] * N_FRAUD
    fraud = coordinates(result)[N_LEGIT:]
    assert (fraud[:CLUSTER_SIZE, 0] > 0).all() and (fraud[CLUSTER_SIZE:, 0] < 0).all()
    assert len(result.synthetic) == int(N_LEGIT * 0.5 - N_FRAUD)  # imblearn's float rule
    assert result.decision_grid is not None
    assert len(result.decision_grid.x) == len(result.decision_grid.y) == sd.GRID_SIZE
    assert result.request == SmoteDemoRequest() and result.is_example is False


@pytest.mark.parametrize("percent", [100, 200, 300, 400, 500])
def test_synthetic_count_is_minority_times_percent(client: TestClient, percent: int) -> None:
    result = demo(client, oversample_percent=percent)
    assert len(result.synthetic) == N_FRAUD * percent // 100


@pytest.mark.parametrize(
    "body",
    [
        {"k_neighbors": 1, "oversample_percent": 500},
        {"k_neighbors": 5, "oversample_percent": 300, "seed": 7},
        {"k_neighbors": 15, "oversample_percent": 200},
        {"scenario": "overlap", "n_fraud": 40, "k_neighbors": 8, "sampling_strategy": 0.4},
    ],
)
def test_every_synthetic_point_lies_on_its_parent_neighbour_segment(
    client: TestClient, body: dict[str, Any]
) -> None:
    result = demo(client, **body)
    points = coordinates(result)
    neighbours = {item.point: item.neighbors for item in result.fraud_neighbors}
    for s in result.synthetic:
        parent, neighbour = points[s.parent], points[s.neighbor]
        assert result.original[s.parent].label == result.original[s.neighbor].label == 1
        assert s.neighbor in neighbours[s.parent]
        assert 0.0 <= s.gap < 1.0
        np.testing.assert_allclose([s.x, s.y], parent + s.gap * (neighbour - parent), atol=TOL)


@pytest.mark.parametrize(("k", "percent", "seed"), [(5, 300, 42), (12, 200, 3), (1, 100, 0)])
def test_teaching_smote_gives_the_same_points_as_imblearn(
    client: TestClient, k: int, percent: int, seed: int
) -> None:
    result = demo(client, k_neighbors=k, oversample_percent=percent, seed=seed)
    X = coordinates(result)
    y = np.array([p.label for p in result.original])
    n_new = N_FRAUD * percent // 100
    smote = SMOTE(sampling_strategy={1: N_FRAUD + n_new}, k_neighbors=k, random_state=seed)
    X_res, _ = smote.fit_resample(X, y)
    ours = np.array([[s.x, s.y] for s in result.synthetic])
    np.testing.assert_allclose(ours, X_res[len(X) :], atol=TOL)


def test_is_deterministic_and_the_dataset_ignores_the_seed(client: TestClient) -> None:
    body = {"k_neighbors": 4, "oversample_percent": 200, "seed": 11}
    first, second = client.post(DEMO, json=body), client.post(DEMO, json=body)
    assert first.content == second.content
    other = demo(client, **(body | {"seed": 12}))
    same = SmoteDemoResponse.model_validate(first.json())
    assert other.original == same.original  # fixed toy data
    assert other.synthetic != same.synthetic


def test_fraud_neighbors_are_the_k_nearest_other_fraud_points(client: TestClient) -> None:
    k = 6
    result = demo(client, k_neighbors=k)
    points = coordinates(result)
    fraud = list(range(N_LEGIT, N_LEGIT + N_FRAUD))
    assert [item.point for item in result.fraud_neighbors] == fraud
    for item in result.fraud_neighbors:
        others = [i for i in fraud if i != item.point]
        distance = {i: float(np.linalg.norm(points[i] - points[item.point])) for i in others}
        assert item.neighbors == sorted(others, key=lambda i: distance[i])[:k]


def test_decision_grid_holds_two_depth_4_trees_on_a_40x40_grid(client: TestClient) -> None:
    seed = 5
    result = demo(client, oversample_percent=200, seed=seed)
    grid = result.decision_grid
    assert grid is not None and grid.model == "DecisionTreeClassifier(max_depth=4)"
    assert len(grid.before) == len(grid.after) == sd.GRID_SIZE
    assert all(len(row) == sd.GRID_SIZE for row in grid.before + grid.after)
    X, y = coordinates(result), np.array([p.label for p in result.original])
    synthetic = np.array([[s.x, s.y] for s in result.synthetic])
    xx, yy = np.meshgrid(grid.x, grid.y)
    cells = np.column_stack([xx.ravel(), yy.ravel()])
    for labels, (X_fit, y_fit) in (
        (grid.before, (X, y)),
        (grid.after, (np.vstack([X, synthetic]), np.r_[y, np.ones(len(synthetic), dtype=int)])),
    ):
        tree = DecisionTreeClassifier(max_depth=4, random_state=seed).fit(X_fit, y_fit)
        assert labels == tree.predict(cells).reshape(sd.GRID_SIZE, sd.GRID_SIZE).tolist()
    assert min(grid.x) < X[:, 0].min() and max(grid.y) > X[:, 1].max()


def test_large_k_links_the_two_clusters_and_lands_in_legit_territory(client: TestClient) -> None:
    small = demo(client, k_neighbors=3, oversample_percent=300)
    large = demo(client, k_neighbors=12, oversample_percent=300)
    link = "can connect distant fraud clusters"
    assert not any(link in w for w in small.warnings)
    assert any(link in w for w in large.warnings)
    assert large.n_synthetic_in_legit_region > small.n_synthetic_in_legit_region
    assert large.n_synthetic_in_legit_region == sum(s.in_legit_region for s in large.synthetic)
    assert any("landed among legitimate points" in w for w in large.warnings)
    assert large.decision_grid is not None
    before, after = large.decision_grid.before, large.decision_grid.after
    assert sum(map(sum, after)) > sum(map(sum, before))  # the bridge points widen the fraud region


def test_in_legit_region_means_most_nearest_original_points_are_legit(client: TestClient) -> None:
    result = demo(client, k_neighbors=12, oversample_percent=500)
    X, y = coordinates(result), np.array([p.label for p in result.original])
    synthetic = np.array([[s.x, s.y] for s in result.synthetic])
    k = sd.LEGIT_REGION_NEIGHBOURS
    nearest = NearestNeighbors(n_neighbors=k).fit(X).kneighbors(synthetic, return_distance=False)
    expected = ((y[nearest] == 0).sum(axis=1) > k / 2).tolist()
    assert [s.in_legit_region for s in result.synthetic] == expected
    assert 0 < sum(expected) < len(expected)


def test_sampling_strategy_rules(client: TestClient) -> None:
    result = demo(client, n_legit=200, n_fraud=10, sampling_strategy=0.3)
    assert len(result.synthetic) == int(200 * 0.3 - 10)
    none = demo(client, n_legit=200, n_fraud=10, sampling_strategy=0.04)
    assert none.synthetic == [] and any("no synthetic points" in w for w in none.warnings)
    both = demo(client, sampling_strategy=0.3, oversample_percent=100)
    assert len(both.synthetic) == N_FRAUD
    assert any("sampling_strategy is ignored" in w for w in both.warnings)
    assert not any("ignored" in w for w in demo(client, oversample_percent=100).warnings)


@pytest.mark.parametrize("scenario", ["single_cluster", "overlap"])
def test_one_cluster_scenarios(client: TestClient, scenario: str) -> None:
    result = demo(client, scenario=scenario, n_legit=150, n_fraud=30, k_neighbors=15)
    assert [p.label for p in result.original].count(1) == 30
    assert not any("connect distant fraud clusters" in w for w in result.warnings)


@pytest.mark.parametrize(
    "body",
    [
        {"k_neighbors": 0},
        {"k_neighbors": 16},
        {"oversample_percent": 150},
        {"oversample_percent": 0},
        {"oversample_percent": 600},
        {"n_fraud": 5, "k_neighbors": 5},
        {"seed": -1},
        {"seed": 2**32},
        {"scenario": "three_clusters"},
        {"lambda": 0.5},
    ],
)
def test_invalid_requests_give_422(client: TestClient, body: dict[str, Any]) -> None:
    response = client.post(DEMO, json=body)
    assert response.status_code == 422, response.text
    error = ErrorResponse.model_validate(response.json()).error
    assert error.code == "validation_error" and error.details


def test_make_toy_data_is_fixed_and_rounded() -> None:
    first = sd.make_toy_data("two_clusters", N_LEGIT, N_FRAUD)
    second = sd.make_toy_data("two_clusters", N_LEGIT, N_FRAUD)
    np.testing.assert_array_equal(first.X, second.X)
    np.testing.assert_array_equal(first.X, np.round(first.X, sd.DATA_DECIMALS))
    assert first.cluster.tolist() == [0] * CLUSTER_SIZE + [1] * CLUSTER_SIZE
    odd = sd.make_toy_data("two_clusters", 10, 5)
    assert odd.cluster.tolist() == [0, 0, 0, 1, 1] and odd.y.tolist() == [0] * 10 + [1] * 5


def test_n_synthetic() -> None:
    assert sd.n_synthetic(SmoteDemoRequest(oversample_percent=300)) == 60
    assert sd.n_synthetic(SmoteDemoRequest(n_legit=100, n_fraud=10, sampling_strategy=0.5)) == 40
    assert sd.n_synthetic(SmoteDemoRequest(n_legit=100, n_fraud=10, sampling_strategy=0.05)) == 0


def test_smote_steps_and_in_legit_region_edge_cases() -> None:
    data = sd.make_toy_data("single_cluster", 50, 8)
    steps = sd.smote_steps(data.X[50:], k=2, n_new=0, seed=0)
    assert steps.points.shape == (0, 2) and steps.neighbours.shape == (8, 2)
    assert sd.in_legit_region(data.X, data.y, steps.points).shape == (0,)
    steps = sd.smote_steps(data.X[50:], k=2, n_new=9, seed=0)
    parents, partners = steps.parents, steps.partners
    assert (steps.neighbours[parents] == partners[:, np.newaxis]).any(axis=1).all()


def test_works_without_artifacts_and_is_rate_limited_to_30_per_minute() -> None:
    limiter.reset()
    limit = int(SMOTE_DEMO_LIMIT.split("/")[0])
    with TestClient(create_app(settings(rate_limit_enabled=True, examples_dir="missing"))) as c:
        assert c.get(f"{API_PREFIX}/health").json()["status"] == "degraded"
        body = {"n_legit": 20, "n_fraud": 4, "k_neighbors": 1}
        assert {c.post(DEMO, json=body).status_code for _ in range(limit)} == {200}
        response = c.post(DEMO, json=body)
    assert response.status_code == 429
    assert ErrorResponse.model_validate(response.json()).error.code == "rate_limited"
