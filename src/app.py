from collections.abc import Callable
from typing import Annotated

import uvicorn
from fastapi import Depends, FastAPI, Request, status
from fastapi.responses import JSONResponse, Response

from .config import CONFIGS
from .llm import LLMService
from .logging import logger

app = FastAPI()


class PATHS:
    health_check = "/health/"


@app.middleware("http")
async def handling_exception(request: Request, call_next: Callable) -> Response:
    try:
        return await call_next(request)
    except Exception as e:
        logger.exception(str(e))
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, content={"detail": f"{e.__class__.__name__}: {e.args}"}
        )


def get_llm_service() -> LLMService:
    return LLMService()


@app.get(path=PATHS.health_check)
async def health_check(
    llm_service: Annotated[LLMService, Depends(get_llm_service)],
) -> Response:
    return JSONResponse(content={"status": "OK"})


def main() -> None:
    uvicorn.run(app, **CONFIGS.uvicorn.model_dump())


if __name__ == "__main__":
    main()
