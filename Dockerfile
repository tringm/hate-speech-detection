FROM python:3.11.7

WORKDIR /app

RUN pip install --upgrade pip && pip install poetry==1.7.1

COPY pyproject.toml .
COPY poetry.toml .
COPY poetry.lock .

RUN poetry config virtualenvs.create false
RUN poetry install --only main --no-root

COPY src ./src
COPY settings.toml .
RUN mkdir "models"

CMD ["python", "-m", "src.app"]
