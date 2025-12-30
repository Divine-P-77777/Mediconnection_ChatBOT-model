FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render injects PORT automatically
EXPOSE 50001

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-50001}"]


# docker build -t dynamicphillic/mediconnectionchatbotapi:v1.0 .
# docker push dynamicphillic/mediconnectionchatbotapi:v1.0
