from typing import Any, TypeVar

from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar, SchemaConverter
from pydantic import BaseModel

from .config import CONFIGS, MODEL_DIR_PATH
from .logging import get_logger
from .prompts import HATE_SPEECH_DETECTION_PROMPT, PromptTemplate

LOGGER = get_logger("LLM")

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


class LLMError(Exception):
    pass


class LLMService:
    def __init__(self) -> None:
        cfg = CONFIGS.llm
        try:
            self.model = Llama(model_path=str(MODEL_DIR_PATH / cfg.model_name), **cfg.model_configs)
        except Exception as e:
            LOGGER.exception(
                "Failed to initiate model `%s` with configs `%s`: %s", cfg.model_name, cfg.model_configs, e
            )
        self.default_prompt_configs = cfg.prompt_configs

    def run(
        self,
        prompt: PromptTemplate,
        prompt_inputs: dict,
        **kwargs: Any,
    ) -> str:
        formatted_prompt = prompt.format_inputs(inputs=prompt_inputs)
        completion_kwargs = {**self.default_prompt_configs, **kwargs, "prompt": formatted_prompt}
        try:
            llm_out = self.model.create_completion(**completion_kwargs)
        except Exception as e:
            LOGGER.exception("Failed to get llm outputs with `%s`: %s", completion_kwargs, e)
            raise LLMError() from e
        try:
            result = llm_out["choices"][0]["text"]
        except Exception as e:
            LOGGER.exception("Failed to parse llm output `%s`: %s", llm_out, e)
            raise LLMError() from e
        return result  # type: ignore

    def run_parse_model(
        self, prompt: PromptTemplate, prompt_inputs: dict, output_model: type[PydanticModel], **kwargs: Any
    ) -> PydanticModel:
        grammar = pydantic_model_to_llama_grammar(model=output_model)
        llm_out = self.run(prompt=prompt, prompt_inputs=prompt_inputs, grammar=grammar, **kwargs)
        try:
            return output_model.model_validate_json(llm_out)
        except Exception as e:
            LOGGER.exception("Failed to parse LLM output to `%s`: %s", output_model.__name__, e)
            raise LLMError() from e


def pydantic_model_to_llama_grammar(model: type[BaseModel]) -> LlamaGrammar:
    try:
        converter = SchemaConverter({})
        converter.visit(model.model_json_schema(), "")
        return LlamaGrammar.from_string(grammar=converter.format_grammar(), verbose=False)
    except Exception as e:
        LOGGER.exception("Failed to generate LlamaGrammar from `%s`: %s", model.__name__, e)
        raise LLMError() from e


class HateSpeechDetectionOutput(BaseModel):
    is_hate_speech: bool
    target_group: list[str]
    reasoning: str


def detect_hate_speech(llm: LLMService, text: str) -> HateSpeechDetectionOutput:
    return llm.run_parse_model(
        prompt=HATE_SPEECH_DETECTION_PROMPT, prompt_inputs={"text": text}, output_model=HateSpeechDetectionOutput
    )
