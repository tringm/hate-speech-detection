import pytest

from src.llm import LLMService, detect_hate_speech


@pytest.mark.parametrize(
    "text, is_hate_speech",
    [
        (
            "We have enough problems in the world. Stop hating on each other",
            False,
        ),
        ("Speak English in America ching chong", True),
    ],
)
def test_detect_hate_speech_simple(llm_service: LLMService, text: str, is_hate_speech: bool) -> None:
    res = detect_hate_speech(llm=llm_service, text=text)
    assert res.is_hate_speech == is_hate_speech, f"Expected is_hate_speech {is_hate_speech}. Got {res}"
    if is_hate_speech:
        assert res.target_group, f"Expected identify target_group. Got {res.target_group}"
    else:
        assert not res.target_group, f"Expected no target group. Got {res.target_group}"
