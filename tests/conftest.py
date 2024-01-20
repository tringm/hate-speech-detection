import os
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import Client
from sqlalchemy import pool
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.app import app, get_llm_service
from src.config import CONFIGS
from src.llm import LLMService
from tests import TEST_DIR_PATH, TEST_OUTPUTS_DIR_PATH


class _FLAGS:
    run_evaluation_tests = "--run-eval"
    run_comparison = "--output-diff"
    run_integration_tests = "--run-integration"


def pytest_addoption(parser) -> None:  # type: ignore
    parser.addoption(_FLAGS.run_evaluation_tests, action="store_true", help="run evaluation tests")
    parser.addoption(_FLAGS.run_comparison, action="store_true", help="show changes of test output file")
    parser.addoption(_FLAGS.run_integration_tests, action="store_true", help="run integration tests")


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    run_eval_tests = config.getoption(_FLAGS.run_evaluation_tests)

    skip_eval_tests = pytest.mark.skip(reason=f"require {_FLAGS.run_evaluation_tests} to run")

    for item in items:
        if "evaluation" in item.keywords and not run_eval_tests:
            item.add_marker(skip_eval_tests)

    run_db_integration_tests = config.getoption(_FLAGS.run_integration_tests)
    skip_integration_tests = pytest.mark.skip(reason=f"require {_FLAGS.run_integration_tests} to run")
    for item in items:
        if not run_db_integration_tests and ("alembic" in item.keywords or "integration" in item.keywords):
            item.add_marker(skip_integration_tests)


@pytest.fixture(scope="session")
def llm_service() -> LLMService:
    return LLMService()


@pytest.fixture
def test_case_out_dir(request: pytest.FixtureRequest) -> Path:
    test_case_path = request.path.parent / request.path.stem
    return TEST_OUTPUTS_DIR_PATH.joinpath(test_case_path.relative_to(TEST_DIR_PATH))


@pytest.fixture
def test_case_out_file(request: pytest.FixtureRequest, test_case_out_dir: Path) -> Path:
    test_case_out_dir.mkdir(parents=True, exist_ok=True)
    return test_case_out_dir / f"{request.node.name}_out.txt"


@pytest.fixture(autouse=True)
def run_comparison(
    request: pytest.FixtureRequest,
    test_case_out_file: Path,
) -> None:
    comp_enabled = request.config.getoption(_FLAGS.run_comparison)
    if comp_enabled and test_case_out_file.exists():
        request.addfinalizer(finalizer=lambda: os.system(f"git difftool {test_case_out_file}"))  # noqa: S605


@pytest.fixture(scope="session")
def application(llm_service: LLMService) -> FastAPI:
    app.dependency_overrides = {
        get_llm_service: lambda: llm_service,
    }
    return app


@pytest.fixture(scope="session")
def client(application: FastAPI) -> Client:
    return Client(base_url=_get_required_env_var(env_var="TEST_API_URL"), timeout=60.0)


@pytest.fixture(scope="session")
def db_engine() -> AsyncEngine:
    return create_async_engine(url=CONFIGS.db.async_conn_str, future=True, poolclass=pool.NullPool)


@pytest.fixture
def alembic_engine(db_engine: AsyncEngine) -> AsyncEngine:
    return db_engine


@pytest_asyncio.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncSession:
    async_session = sessionmaker(bind=db_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session


def _get_required_env_var(env_var: str) -> str:
    val = os.getenv(env_var)
    if not val:
        raise KeyError(f"Missing required env var `{env_var}`")
    return val
