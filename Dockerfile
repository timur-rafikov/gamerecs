# Этап 1: собираем annoy без -march=native, иначе на другом процессоре будет SIGILL
FROM python:3.12-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN pip download annoy==1.17.3 --no-binary :all: --no-deps -d . \
    && tar xzf annoy-1.17.3.tar.gz \
    && sed -i 's/-march=native/-march=x86-64/' annoy-1.17.3/setup.py \
    && pip wheel ./annoy-1.17.3 --no-deps -w /wheels

# Этап 2: рабочий образ без компилятора
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir $(grep -v '^annoy' requirements.txt) /wheels/annoy-*.whl \
    && rm -rf /wheels

RUN useradd -m app
COPY --chown=app:app app.py meta.parquet games.ann ./
COPY --chown=app:app static ./static
USER app

# порт берём из $PORT (Render и др. задают его сами), по умолчанию 8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
