from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from llama_cpp import Llama
from llama_cpp.llama_grammar import LlamaGrammar, SchemaConverter
from llama_cpp.llama_types import CreateCompletionResponse
from pydantic import BaseModel

from src.config import CONFIGS, MODEL_DIR_PATH
from src.db.models import LLMRun
from src.logging import get_logger

from .prompts import PromptTemplate

LOGGER = get_logger("LLM")

PydanticModel = TypeVar("PydanticModel", bound=BaseModel)


class LLMError(Exception):
    pass


class LLMRunParsedModel(LLMRun, Generic[PydanticModel]):
    parsed_output: PydanticModel | None = None


class LLMService:
    def __init__(self, tracking: bool = False) -> None:
        llm_cfg = CONFIGS.llm
        self.model_name = llm_cfg.model_name
        self.model_configs = llm_cfg.model_configs
        try:
            self.model = Llama(model_path=str(MODEL_DIR_PATH / self.model_name), **self.model_configs)
        except Exception as e:
            msg = "Failed to initiate model"
            LOGGER.exception("%s `%s` with configs `%s`: %s", msg, llm_cfg.model_name, llm_cfg.model_configs, e)
            raise LLMError(msg) from e
        self.default_run_configs = llm_cfg.run_configs

    def _llm_completion(self, run_configs: dict) -> CreateCompletionResponse:
        try:
            return self.model.create_completion(**run_configs)
        except Exception as e:
            msg = "Failed to get llm outputs"
            LOGGER.exception("%s with `%s`: %s", msg, run_configs, e)
            raise LLMError(msg) from e

    def run_completion(
        self,
        prompt: PromptTemplate,
        prompt_inputs: dict,
        **kwargs: Any,
    ) -> LLMRun:
        llm_run = LLMRun(
            success=True,
            prompt=prompt.format_inputs(inputs=prompt_inputs),
            llm_model_name=self.model_name,
            llm_model_configs=self.model_configs,
            run_configs={**self.default_run_configs, **kwargs},
            start_run=datetime.now(tz=timezone.utc),
        )
        try:
            llm_out = self.model.create_completion(prompt=llm_run.prompt, **llm_run.run_configs)
            llm_run.llm_outputs = llm_out
        except Exception as e:
            err_msg = f"Failed to get run llm completion: {e}"
            LOGGER.exception(err_msg)
            llm_run.success = False
            llm_run.error = err_msg
        finally:
            llm_run.end_run = datetime.now(tz=timezone.utc)
        return llm_run

    def run_completion_parse_model(
        self, prompt: PromptTemplate, prompt_inputs: dict, output_model: type[PydanticModel], **kwargs: Any
    ) -> LLMRunParsedModel:
        grammar = pydantic_model_to_llama_grammar(model=output_model)
        llm_run = self.run_completion(prompt=prompt, prompt_inputs=prompt_inputs, grammar=grammar, **kwargs)
        if not llm_run.success or not llm_run.llm_outputs:
            return LLMRunParsedModel(**llm_run.model_dump())
        try:
            return LLMRunParsedModel(
                parsed_output=output_model.model_validate_json(llm_run.llm_outputs["choices"][0]["text"]),
                **llm_run.model_dump(),
            )
        except Exception as e:
            err_msg = f"Failed to parse LLM output to {output_model.__name__}: {e}"
            LOGGER.exception(err_msg)
            return LLMRunParsedModel(**{**llm_run.model_dump(), "success": False, "error": err_msg})


def pydantic_model_to_llama_grammar(model: type[BaseModel]) -> LlamaGrammar:
    try:
        converter = SchemaConverter({})
        converter.visit(model.model_json_schema(), "")
        return LlamaGrammar.from_string(grammar=converter.format_grammar(), verbose=False)
    except Exception as e:
        msg = "Failed to generate LlamaGrammar"
        LOGGER.exception("%s from `%s`: %s", msg, model.__name__, e)
        raise LLMError(msg) from e
