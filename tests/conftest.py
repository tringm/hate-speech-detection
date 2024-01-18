import pytest

from src.llm import LLMService


class _FLAGS:
    run_evaluation_tests = "--run-eval"


def pytest_addoption(parser) -> None:  # type: ignore
    parser.addoption(_FLAGS.run_evaluation_tests, action="store_true", help="run evaluation tests")


def pytest_collection_modifyitems(session: pytest.Session, config: pytest.Config, items: list[pytest.Item]) -> None:
    run_eval_tests = config.getoption(_FLAGS.run_evaluation_tests)

    skip_eval_tests = pytest.mark.skip(reason=f"require {_FLAGS.run_evaluation_tests} to run")

    for item in items:
        if "evaluation" in item.keywords and not run_eval_tests:
            item.add_marker(skip_eval_tests)


@pytest.fixture(scope="session")
def llm_service() -> LLMService:
    return LLMService()
