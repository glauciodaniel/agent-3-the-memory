# Stage 1: builder
FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH"

# Non-root user (optional: uncomment to run as appuser)
# RUN addgroup --system app && adduser --system --ingroup app appuser
# USER appuser

COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY config/ ./config/
COPY prompts/ ./prompts/

CMD ["python", "-m", "src.main"]
