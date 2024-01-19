from pydantic import BaseModel

from src.llm import LLMService

from .prompts import HATE_SPEECH_DETECTION_PROMPT, INPUT_TEXT_KEY


class DetectHateSpeechResponse(BaseModel):
    is_hate_speech: bool
    target_of_hate: list[str]
    reasoning: str


def llm_detect_hate_speech(llm: LLMService, text: str) -> DetectHateSpeechResponse:
    return llm.run_parse_model(
        prompt=HATE_SPEECH_DETECTION_PROMPT,
        prompt_inputs={INPUT_TEXT_KEY: text},
        output_model=DetectHateSpeechResponse,
    )
