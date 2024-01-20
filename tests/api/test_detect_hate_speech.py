import pytest
from httpx import Client, Response, codes

from src.app import DetectHateSpeechRequest, DetectHateSpeechResponse, PATHS
from tests.llm.test_detect_hate_speech import EXAMPLE_HATE_SPEECH, EXAMPLE_NOT_HATE_SPEECH


@pytest.mark.parametrize(
    "req_json",
    [
        {},
        {"textt": "This is a hate speech"},
    ],
)
def test_detect_hate_speech_endpoint_bad_request(client: Client, req_json: dict) -> None:
    resp = client.post(url=PATHS.detect_hate_speech, json=req_json)
    assert resp.status_code == codes.BAD_REQUEST


def _assert_hate_speech_response(resp: Response, is_hate_speech: bool) -> None:
    assert resp.status_code == codes.OK, f"Expected {codes.OK}, Got {resp}"
    resp = DetectHateSpeechResponse.model_validate(resp.json())
    if is_hate_speech:
        assert resp.is_hate_speech, "Expected identify as hate speech"
        assert resp.target_of_hate, "Expected identify target of hate"
        assert resp.reasoning, "Expected return reasoning"
    else:
        assert not resp.is_hate_speech, "Expected identify as not hate speech"
        assert not resp.target_of_hate, "Expected no target of hate"


@pytest.mark.parametrize(
    "text, is_hate_speech",
    [
        (EXAMPLE_NOT_HATE_SPEECH, False),
        (EXAMPLE_HATE_SPEECH, True),
    ],
)
def test_detect_hate_speech_endpoint(client: Client, text: str, is_hate_speech: bool) -> None:
    resp = client.post(url=PATHS.detect_hate_speech, json=DetectHateSpeechRequest(text=text).model_dump())
    _assert_hate_speech_response(resp=resp, is_hate_speech=is_hate_speech)
