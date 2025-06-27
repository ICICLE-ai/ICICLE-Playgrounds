FROM python:3.13-slim-bookworm
LABEL author="guzman109"
RUN useradd -ms /bin/bash guzman109
RUN mkdir /app && chown guzman109 /app

USER icicle
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY ./uv.lock .
COPY ./pyproject.toml .

RUN #uv sync --frozen --no-cache

#COPY ./bioclip_api ./bioclip_api

ENV PYTHONPATH="${PYTHONPATH}:/app/bioclip_api"
ENTRYPOINT ["uv", "run", "bioclip_api/server.py"]