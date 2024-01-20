from collections.abc import Callable
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel

from src.db.models import DetectHateSpeechResponse
from src.llm import LLMService
from src.llm.detect_hate_speech import LLMDetectHateSpeechResult, llm_detect_hate_speech
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
) -> Response:
    return JSONResponse(content={"status": "OK"})


@app.post(path=PATHS.detect_hate_speech)
async def detect_hate_speech(
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
    req: DetectHateSpeechRequest,
) -> DetectHateSpeechResponse:
    llm_res = llm_detect_hate_speech(llm=llm_service, text=req.text)
    resp = DetectHateSpeechResponse(text=req.text, is_hate_speech=llm_res.is_hate_speech, target_of_hate=llm_res.target_of_hate, reasoning=llm_res.reasoning)
    return resp


def main() -> None:
    uvicorn.run(app, **CONFIGS.uvicorn.model_dump())


if __name__ == "__main__":
    main()
