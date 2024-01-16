from dynaconf import Dynaconf
from pydantic import BaseModel

APPLICATION_NAME = "hate-speech-detection"

settings = Dynaconf(
    envvar_prefix="DYNACONF",
    settings_files=["settings.toml"],
    environments=True,
)


class RootConfig(BaseModel):
    log_level: str


CONFIGS = RootConfig.model_validate({k.lower(): v for k, v in settings.as_dict().items()})
