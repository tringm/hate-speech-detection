import uuid as uuid_pkg

from pydantic import AwareDatetime, BaseModel
from sqlalchemy import text
from sqlmodel import ARRAY, JSON, Column, DateTime, Field, SQLModel, String


class BaseSQLModel(SQLModel):
    pass


class UUIDModelMixin(BaseSQLModel):
    uuid: uuid_pkg.UUID = Field(
        default_factory=uuid_pkg.uuid4,
        primary_key=True,
        index=True,
        nullable=False,
        sa_column_kwargs={"server_default": text("gen_random_uuid()"), "unique": True},
    )


class LLMRun(BaseModel):
    prompt: str
    llm_model_name: str
    success: bool
    llm_model_configs: dict
    run_configs: dict
    llm_outputs: dict | None = None
    error: str | None = None
    start_run: AwareDatetime | None = None
    end_run: AwareDatetime | None = None


class LLMRunSQLModel(UUIDModelMixin, LLMRun, table=True):
    __tablename__ = "llm_run"

    llm_model_configs: dict = Field(sa_column=Column(JSON), default_factory=dict)
    run_configs: dict = Field(sa_column=Column(JSON), default_factory=dict)
    llm_outputs: dict = Field(sa_column=Column(JSON), default_factory=dict)
    start_run: AwareDatetime | None = Field(sa_column=Column(DateTime(timezone=True)))
    end_run: AwareDatetime | None = Field(sa_column=Column(DateTime(timezone=True)))


class DetectHateSpeechResult(BaseModel):
    is_hate_speech: bool
    target_of_hate: list[str]
    reasoning: str


class DetectHateSpeech(DetectHateSpeechResult):
    text: str


class DetectHateSpeechSQLModel(UUIDModelMixin, DetectHateSpeech, table=True):
    __tablename__ = "detect_hate_speech_result"

    target_of_hate: list[str] = Field(sa_column=Column(ARRAY(String)))
