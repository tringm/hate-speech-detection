from typing import Any

from pydantic import BaseModel

from .logging import get_logger

LOGGER = get_logger(__name__)


class PromptTemplate(BaseModel):
    template: str
    input_keys: list[str]

    def format_inputs(self, inputs: dict[str, Any]) -> str:
        input_keys = set(inputs.keys())
        prompt_input_keys = set(self.input_keys)
        if not prompt_input_keys.issubset(input_keys):
            raise KeyError(f"Missing required input keys: {prompt_input_keys - input_keys}")
        try:
            return self.template.format(**{k: inputs[k] for k in self.input_keys})
        except Exception as e:
            LOGGER.exception("Failed to format prompt %s with value %s: %s", self.template, inputs, e)
            raise
