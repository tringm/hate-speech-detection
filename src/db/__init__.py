import json

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.config import CONFIGS


def json_skip_non_serializable_serializer(obj: dict) -> str:
    return json.dumps(obj, default=lambda o: f"{type(o).__qualname__}")


async_engine = create_async_engine(
    CONFIGS.db.async_conn_str, future=True, json_serializer=json_skip_non_serializable_serializer
)


async def get_async_session() -> AsyncSession:
    async_session = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)
    async with async_session() as session:
        yield session
