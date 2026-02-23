FROM ultralytics/ultralytics:latest

# Install additional Python packages
RUN pip install --no-cache-dir redis httpx

WORKDIR /app
COPY detector.py .

CMD ["python", "detector.py"]
