FROM python:3.11-slim

# Open3D needs these shared libraries even in headless mode.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libgomp1 libx11-6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY pcinspect ./pcinspect
RUN pip install --no-cache-dir . faiss-cpu

# Bake the fitted banks into the image (built with scripts/build_bank.py).
COPY models ./models
ENV PCINSPECT_MODELS=/app/models

EXPOSE 8000
CMD ["uvicorn", "pcinspect.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
