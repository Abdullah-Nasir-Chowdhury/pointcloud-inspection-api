FROM python:3.11-slim

# Open3D needs these shared libraries even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libgomp1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY pcinspect ./pcinspect
RUN pip install --no-cache-dir . faiss-cpu gradio

# Fitted banks (5 MB per category, built with scripts/build_bank.py) and the demo UI.
COPY models ./models
COPY demo ./demo
ENV PCINSPECT_MODELS=/app/models
ENV PCINSPECT_DEMO=/app/demo
ENV PORT=8000

EXPOSE 8000
# Cloud Run and most PaaS inject $PORT; default to 8000 locally.
CMD ["sh", "-c", "uvicorn pcinspect.api.app:app --host 0.0.0.0 --port ${PORT}"]
