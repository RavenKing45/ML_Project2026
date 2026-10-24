"""Fixtures of the prediction tests (Prompt 13); the models themselves are built by
``real_models.py`` in this folder."""

from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from real_models import RealArtifacts, build_real_artifacts, real_settings

from backend.app.main import create_app
from backend.app.ratelimit import limiter
from backend.app.settings import Settings
from fraudstudy.experiment import Configs


@pytest.fixture(scope="session")
def real_artifacts(tmp_path_factory: pytest.TempPathFactory, test_cfgs: Configs) -> RealArtifacts:
    """The small real models and their files, built once per session."""
    return build_real_artifacts(tmp_path_factory.mktemp("real_artifacts"), test_cfgs)


@pytest.fixture(scope="session")
def make_real_settings(real_artifacts: RealArtifacts) -> Callable[..., Settings]:
    """A factory of :func:`real_settings` for the session's tree (keyword arguments win)."""

    def make(**overrides: Any) -> Settings:
        return real_settings(real_artifacts, **overrides)

    return make


@pytest.fixture(scope="module")
def real_client(make_real_settings: Callable[..., Settings]) -> Iterator[TestClient]:
    """A started app on the real tree, per test module.

    Not per session: the app starts in TestClient's portal thread, and LightGBM's OpenMP thread
    team created there lives as long as that thread. Kept open into the training tests, the extra
    OpenMP threads make libgomp stop spin-waiting and LightGBM fits several times slower.
    """
    with TestClient(create_app(make_real_settings())) as test_client:
        yield test_client


@pytest.fixture
def rate_limits_off() -> Iterator[None]:
    """Switch the shared slowapi limiter off and clear its counters around a test.

    The limiter is module-level, so an app built with limits on (in a rate-limit test, or in
    tests/backend/test_core.py) turns them on for every app; this keeps the other tests from
    reaching a limit.
    """
    limiter.reset()
    limiter.enabled = False
    yield
    limiter.reset()
