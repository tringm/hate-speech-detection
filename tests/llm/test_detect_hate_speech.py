from pathlib import Path

import pytest
from sklearn import metrics

from src.llm import LLMService
from src.llm.detect_hate_speech import llm_detect_hate_speech
from src.logging import logger
from tests.data.hatexplain import load_hatexplain_detection_cases

EXAMPLE_NOT_HATE_SPEECH = "We have enough problems in the world. Stop hating on each other"
EXAMPLE_HATE_SPEECH = "Speak English in America ching chong"


@pytest.mark.parametrize(
    "text, is_hate_speech",
    [
        (EXAMPLE_NOT_HATE_SPEECH, False),
        (EXAMPLE_HATE_SPEECH, True),
    ],
)
def test_detect_hate_speech_simple(llm_service: LLMService, text: str, is_hate_speech: bool) -> None:
    res = llm_detect_hate_speech(llm=llm_service, text=text)
    assert res.is_hate_speech == is_hate_speech, f"Expected is_hate_speech {is_hate_speech}. Got {res}"
    if is_hate_speech:
        assert res.target_of_hate, f"Expected identify target. Got {res.target_of_hate}"
    else:
        assert not res.target_of_hate, f"Expected no target. Got {res.target_of_hate}"


@pytest.mark.evaluation
def test_evaluate_detect_hate_speech_hatexplain(llm_service: LLMService, test_case_out_file: Path) -> None:
    y_true, y_pred = [], []

    with test_case_out_file.open(mode="w") as f:
        for case in load_hatexplain_detection_cases():
            f.write("Test case:\n")
            f.write(case.model_dump_json(indent=2, exclude={"uuid"}) + "\n")
            y_true.append(case.is_hate_speech)
            try:
                res = llm_detect_hate_speech(llm=llm_service, text=case.text)
            except Exception as e:
                logger.exception(e)
                y_pred.append(-1)
                continue
            y_pred.append(res.is_hate_speech)
            f.write("Result:\n")
            f.write(res.model_dump_json(indent=2) + "\n")
            f.write("-" * 10 + "\n")

        precision, recall, f_score, support = metrics.precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, labels=[True, False]
        )
        for idx, case_name in enumerate(["Hate cases", "Not Hate cases"]):
            f.write(
                f"""{case_name}:
- No. case: {support[idx]}
- Precision: {precision[idx]}
- Recall: {recall[idx]}
- F-score: {f_score[idx]}
"""
            )
