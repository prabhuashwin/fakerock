FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    MOCK_BEDROCK_HOST=0.0.0.0 \
    MOCK_BEDROCK_LOG_DIR=/data

RUN useradd --system --create-home --uid 10001 mock \
    && mkdir /data && chmod 777 /data

WORKDIR /app
COPY mock_bedrock/ /app/mock_bedrock/

USER mock
VOLUME /data
# Change the port with -e MOCK_BEDROCK_PORT=... (not --port): the healthcheck reads that variable.
EXPOSE 52999

HEALTHCHECK --interval=10s --timeout=3s --start-period=3s --retries=3 \
    CMD ["python", "-c", "import os,urllib.request as u; u.urlopen('http://127.0.0.1:%s/health' % os.environ.get('MOCK_BEDROCK_PORT','52999'), timeout=2)"]

ENTRYPOINT ["python", "-m", "mock_bedrock"]
