from pathlib import Path

from dynaconf import Dynaconf
from pydantic import BaseModel, ConfigDict

SRC_PATH = Path(__file__).parent.resolve()
PROJECT_ROOT_PATH = SRC_PATH.parent
MODEL_DIR_PATH = PROJECT_ROOT_PATH / "models"

APPLICATION_NAME = "hate-speech-detection"

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml"],
    environments=True,
)


class LLMConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_name: str
    model_configs: dict
    run_configs: dict


class UvicornConfig(BaseModel):
    host: str
    port: int


class DBConfig(BaseModel):
    async_conn_str: str


class RootConfig(BaseModel):
    log_level: str

    llm: LLMConfig
    uvicorn: UvicornConfig
    db: DBConfig


CONFIGS = RootConfig.model_validate({k.lower(): v for k, v in settings.as_dict().items()})
