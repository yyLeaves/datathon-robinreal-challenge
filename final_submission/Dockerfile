FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv pip install --system .

COPY app ./app
COPY apps_sdk ./apps_sdk
COPY raw_data ./raw_data
COPY README.md ./

ENV LISTINGS_RAW_DATA_DIR=/app/raw_data
ENV LISTINGS_DB_PATH=/data/listings.db

ENV CLAUDE_FAST_MODEL=claude-haiku-4-5
ENV CLAUDE_SMART_MODEL=claude-opus-4-7
ENV CLAUDE_DEFAULT_MAX_TOKENS=1024
ENV CLAUDE_EXTRACTION_MAX_TOKENS=512
ENV CLAUDE_RANKING_MAX_TOKENS=4096

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
