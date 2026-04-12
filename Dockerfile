FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir ".[telegram,quant]"

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin/nct /usr/local/bin/nct
COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

RUN useradd --create-home --shell /bin/bash trader \
    && mkdir -p /app/logs /app/data /app/data/models \
    && chown -R trader:trader /app/logs /app/data

USER trader

HEALTHCHECK --interval=60s --timeout=10s --retries=3 \
    CMD python scripts/healthcheck.py

ENTRYPOINT ["nct"]
