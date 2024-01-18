import pytest

from src.llm import LLMService


@pytest.fixture(scope="session")
def llm_service() -> LLMService:
    return LLMService()
