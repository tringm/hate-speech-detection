from typing import Any, TypeVar

from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar, SchemaConverter
from pydantic import BaseModel

from src.config import CONFIGS, MODEL_DIR_PATH
from src.logging import get_logger

from .prompts import PromptTemplate

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
            msg = "Failed to initiate model"
            LOGGER.exception("%s `%s` with configs `%s`: %s", msg, cfg.model_name, cfg.model_configs, e)
            raise LLMError(msg) from e
        self.default_run_configs = cfg.run_configs

    def run(
        self,
        prompt: PromptTemplate,
        prompt_inputs: dict,
        **kwargs: Any,
    ) -> str:
        formatted_prompt = prompt.format_inputs(inputs=prompt_inputs)
        run_configs = {**self.default_run_configs, **kwargs, "prompt": formatted_prompt}
        try:
            llm_out = self.model.create_completion(**run_configs)
        except Exception as e:
            msg = "Failed to get llm outputs"
            LOGGER.exception("%s with `%s`: %s", msg, run_configs, e)
            raise LLMError(msg) from e
        try:
            result = llm_out["choices"][0]["text"]
        except Exception as e:
            msg = "Failed to parse llm output"
            LOGGER.exception("%s `%s`: %s", msg, llm_out, e)
            raise LLMError(msg) from e
        return result  # type: ignore

    def run_parse_model(
        self, prompt: PromptTemplate, prompt_inputs: dict, output_model: type[PydanticModel], **kwargs: Any
    ) -> PydanticModel:
        grammar = pydantic_model_to_llama_grammar(model=output_model)
        llm_out = self.run(prompt=prompt, prompt_inputs=prompt_inputs, grammar=grammar, **kwargs)
        try:
            return output_model.model_validate_json(llm_out)
        except Exception as e:
            msg = "Failed to parse LLM output"
            LOGGER.exception("%s to `%s`: %s", msg, output_model.__name__, e)
            raise LLMError(msg) from e


def pydantic_model_to_llama_grammar(model: type[BaseModel]) -> LlamaGrammar:
    try:
        converter = SchemaConverter({})
        converter.visit(model.model_json_schema(), "")
        return LlamaGrammar.from_string(grammar=converter.format_grammar(), verbose=False)
    except Exception as e:
        msg = "Failed to generate LlamaGrammar"
        LOGGER.exception("%s from `%s`: %s", msg, model.__name__, e)
        raise LLMError(msg) from e
