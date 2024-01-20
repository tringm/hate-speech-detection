import pytest
from httpx import Client, Response, codes
from sqlalchemy.ext.asyncio import AsyncSession

from src.app import PATHS, DetectHateSpeechRequest, DetectHateSpeechResult
from tests.llm.test_detect_hate_speech import EXAMPLE_HATE_SPEECH, EXAMPLE_NOT_HATE_SPEECH


@pytest.mark.integration
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


def _assert_hate_speech_response(resp: Response, is_hate_speech: bool) -> DetectHateSpeechResult:
    assert resp.status_code == codes.OK, f"Expected {codes.OK}, Got {resp}"
    resp = DetectHateSpeechResult.model_validate(resp.json())
    if is_hate_speech:
        assert resp.is_hate_speech, "Expected identify as hate speech"
        assert resp.target_of_hate, "Expected identify target of hate"
        assert resp.reasoning, "Expected return reasoning"
    else:
        assert not resp.is_hate_speech, "Expected identify as not hate speech"
        assert not resp.target_of_hate, "Expected no target of hate"
    return resp  # type: ignore


@pytest.mark.integration
@pytest.mark.parametrize(
    "text, is_hate_speech",
    [
        (EXAMPLE_NOT_HATE_SPEECH, False),
        (EXAMPLE_HATE_SPEECH, True),
    ],
)
@pytest.mark.asyncio
async def test_detect_hate_speech_endpoint(
    client: Client, db_session: AsyncSession, text: str, is_hate_speech: bool
) -> None:
    resp = client.post(url=PATHS.detect_hate_speech, json=DetectHateSpeechRequest(text=text).model_dump())
    resp_model = _assert_hate_speech_response(resp=resp, is_hate_speech=is_hate_speech)
    db_resp = await db_session.get(DetectHateSpeechResult, resp_model.uuid)
    assert db_resp, f"Expected a DetectHateSpeechResponse with uuid {resp_model.uuid} in DB"
