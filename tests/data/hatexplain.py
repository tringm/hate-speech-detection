import gzip
from collections.abc import Iterator
from enum import Enum

from pydantic import BaseModel, Field

from src.detect_hate_speech import DetectHateSpeechResponse
from src.logging import get_logger
from tests import DATA_DIR_PATH

HATEXPLAIN_DATA_FILE = DATA_DIR_PATH / "hatexplain.ndjson.gz"
HATEXPLAIN_DETECTION_CASE_DATA_FILE = DATA_DIR_PATH / "hatexplain_detection_cases.ndjson.gz"

LOGGER = get_logger("Hatexplain")


class HatexplainLabel(Enum):
    normal = "normal"
    offensive = "offensive"
    hate = "hatespeech"


class HatexplainAnnotatorResult(BaseModel):
    label: HatexplainLabel
    target: list[str]


class HatexplainCase(BaseModel):
    post_id: str
    annotator_results: list[HatexplainAnnotatorResult] = Field(alias="annotators")
    annotator_rationales_by_token_idx: list[list[bool]] = Field(alias="rationales")
    post_tokens: list[str]

    @property
    def unanimous_label(self) -> HatexplainLabel | None:
        labels = {res.label for res in self.annotator_results}
        if len(labels) == 1:
            return labels.pop()
        return None

    @property
    def all_targets(self) -> list[str]:
        return list({target for res in self.annotator_results for target in res.target if target and target != "None"})

    @property
    def all_rationale_tokens(self) -> list[str]:
        all_tok_idx = sorted(
            {
                tok_idx
                for rationales in self.annotator_rationales_by_token_idx
                for tok_idx, is_rat in enumerate(rationales)
                if is_rat
            }
        )
        return [self.post_tokens[idx] for idx in all_tok_idx]

    @property
    def text(self) -> str:
        return " ".join(self.post_tokens)


class HateSpeechDetectionCase(DetectHateSpeechResponse):
    text: str


def load_hatexplain_detection_cases() -> Iterator[HateSpeechDetectionCase]:
    with gzip.open(HATEXPLAIN_DETECTION_CASE_DATA_FILE, mode="rt") as in_f:
        for line in in_f:
            try:
                yield HateSpeechDetectionCase.model_validate_json(json_data=line.strip())
            except Exception as e:
                LOGGER.error("Failed to parse HateSpeechDetectionCase %s: %s", line, e)
                continue


def build_hatexplain_detection_cases() -> None:
    """Generate HateSpeechDetectionCase from Hatexplain cases that have unanimous annotated label and save to file"""
    with gzip.open(HATEXPLAIN_DATA_FILE, mode="rt") as in_f, gzip.open(
        HATEXPLAIN_DETECTION_CASE_DATA_FILE, mode="wt"
    ) as u_hate_f:
        for line in in_f:
            try:
                case = HatexplainCase.model_validate_json(json_data=line.strip())
            except Exception as e:
                LOGGER.error("Failed to parse HatexplainCase %s: %s", line, e)
                continue
            if case.unanimous_label:
                u_case = HateSpeechDetectionCase(
                    text=case.text,
                    is_hate_speech=case.unanimous_label != HatexplainLabel.normal,
                    target_of_hate=case.all_targets,
                    reasoning=" ".join(case.all_rationale_tokens),
                )
                u_hate_f.write(u_case.model_dump_json())
                u_hate_f.write("\n")


if __name__ == "__main__":
    build_hatexplain_detection_cases()
