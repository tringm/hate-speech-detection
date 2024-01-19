import pytest

from src.llm.prompts import PromptTemplate

SAMPLE_PROMPT = PromptTemplate(template="Some text: {text}", input_keys=["text"])


def test_prompt_format() -> None:
    text = "Some text"
    formatted_prompt = SAMPLE_PROMPT.format_inputs(inputs={"text": text})
    assert formatted_prompt == f"Some text: {text}"


def test_prompt_format_missing_input() -> None:
    text = "Some text"
    with pytest.raises(KeyError):
        SAMPLE_PROMPT.format_inputs(inputs={"textt": text})
