import os
from pathlib import Path

import pytest

from src.llm import LLMService
from tests import TEST_DIR_PATH, TEST_OUTPUTS_DIR_PATH


class _FLAGS:
    run_evaluation_tests = "--run-eval"
    run_comparison = "--output-diff"


def pytest_addoption(parser) -> None:  # type: ignore
    parser.addoption(_FLAGS.run_evaluation_tests, action="store_true", help="run evaluation tests")
    parser.addoption(_FLAGS.run_comparison, action="store_true", help="show changes of test output file")


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    run_eval_tests = config.getoption(_FLAGS.run_evaluation_tests)

    skip_eval_tests = pytest.mark.skip(reason=f"require {_FLAGS.run_evaluation_tests} to run")

    for item in items:
        if "evaluation" in item.keywords and not run_eval_tests:
            item.add_marker(skip_eval_tests)


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
