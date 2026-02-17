FROM pytorch/pytorch:2.2.2-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart \
    pandas numpy scikit-learn joblib openpyxl \
    pillow matplotlib

EXPOSE 8000
CMD ["uvicorn", "main_api:app", "--host", "0.0.0.0", "--port", "8000"]
