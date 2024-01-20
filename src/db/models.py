import uuid as uuid_pkg

from pydantic import AwareDatetime
from sqlalchemy import text
from sqlmodel import ARRAY, JSON, Column, DateTime, Field, SQLModel, String


class BaseModel(SQLModel):
    pass


class UUIDModelMixin(BaseModel):
    uuid: uuid_pkg.UUID = Field(
        default_factory=uuid_pkg.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
        sa_column_kwargs={"server_default": text("gen_random_uuid()"), "unique": True},
    )


class LLMPromptRun(UUIDModelMixin, table=True):  # type: ignore
    __tablename__ = "llm_prompt_run"

    success: bool
    prompt: str
    llm_model_name: str
    llm_model_configs: dict = Field(sa_column=Column(JSON), default={})
    run_configs: dict = Field(sa_column=Column(JSON), default={})
    llm_outputs: dict = Field(sa_column=Column(JSON), default={})
    start_run: AwareDatetime | None = Field(sa_column=Column(DateTime(timezone=True)))
    end_run: AwareDatetime | None = Field(sa_column=Column(DateTime(timezone=True)))


class DetectHateSpeechResult(UUIDModelMixin, table=True):  # type: ignore
    __tablename__ = "detect_hate_speech_result"

    text: str
    is_hate_speech: bool
    target_of_hate: list[str] = Field(sa_column=Column(ARRAY(String)))
    reasoning: str
