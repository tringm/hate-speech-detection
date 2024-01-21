from collections.abc import Callable
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from src.db import get_async_session
from src.db.models import DetectHateSpeechSQLModel, LLMRunSQLModel
from src.llm import LLMService
from src.llm.detect_hate_speech import llm_detect_hate_speech

from .config import CONFIGS
from .logging import logger

app = FastAPI()


class PATHS:
    health_check = "/health/"
    detect_hate_speech = "/detect_hate_speech/"


class DetectHateSpeechRequest(BaseModel):
    text: str


@app.middleware("http")
async def handling_exception(request: Request, call_next: Callable) -> Response:
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception(str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": f"{e.__class__.__name__}: {e.args}"}
        )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> Response:
    return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"detail": exc.errors()})


def get_llm_service() -> LLMService:
    return LLMService()


@app.get(path=PATHS.health_check)
async def health_check(
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
) -> Response:
    return JSONResponse(content={"status": "OK"})


@app.post(path=PATHS.detect_hate_speech)
async def detect_hate_speech(
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    session: Annotated[AsyncSession, Depends(get_async_session)],
    req: DetectHateSpeechRequest,
) -> DetectHateSpeechSQLModel:
    text = req.text

    llm_run = llm_detect_hate_speech(llm=llm_service, text=text)
    llm_run_sql = LLMRunSQLModel(**llm_run.model_dump())
    session.add(llm_run_sql)
    await session.commit()

    if not llm_run.success or not llm_run.parsed_output:
        await session.commit()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=llm_run.error)

    detect_res = DetectHateSpeechSQLModel(text=text, llm_run_id=llm_run_sql.uuid, **llm_run.parsed_output.model_dump())
    session.add(detect_res)

    await session.commit()

    return detect_res


def main() -> None:
    uvicorn.run(app, **CONFIGS.uvicorn.model_dump())


if __name__ == "__main__":
    main()
