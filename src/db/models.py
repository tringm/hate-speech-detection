import uuid as uuid_pkg

from pydantic import AwareDatetime
from sqlalchemy import text
from sqlmodel import JSON, Column, DateTime, Field, SQLModel


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
    run_configs: dict = Field(sa_column=Column(JSON), default={})
    llm_outputs: dict = Field(sa_column=Column(JSON), default={})
    start_run: AwareDatetime | None = Field(sa_column=Column(DateTime(timezone=True)))
    end_run: AwareDatetime | None = Field(sa_column=Column(DateTime(timezone=True)))
