from src.db.models import DetectHateSpeech
from src.llm import LLMService

from .prompts import HATE_SPEECH_DETECTION_PROMPT, INPUT_TEXT_KEY


def llm_detect_hate_speech(llm: LLMService, text: str) -> DetectHateSpeech:
    return llm.run_parse_model(
        prompt=HATE_SPEECH_DETECTION_PROMPT,
        prompt_inputs={INPUT_TEXT_KEY: text},
        output_model=DetectHateSpeech,
    )
