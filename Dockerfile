# Dockerfile
# One file, two images. The API and the worker install requirements.txt;
# serving/ installs requirements-local.txt, which pulls about 2GB of
# torch. The build argument is what keeps that 2GB out of the image that
# does not need it, which is the operational point of the split
# stated in the only place a deployment can read it.
ARG REQUIREMENTS=requirements.txt

FROM python:3.12-slim
ARG REQUIREMENTS
ARG BUILD_ID=dev
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 BUILD_ID=${BUILD_ID}
WORKDIR /srv

# Requirements first, and alone: this layer is cached until a dependency
# changes, so an ordinary code edit rebuilds in seconds rather than
# re-downloading torch.
COPY requirements.txt requirements-local.txt ./
RUN pip install --no-cache-dir -r ${REQUIREMENTS}

COPY alembic.ini reindex.py ./
COPY app app
COPY serving serving
COPY migrations migrations

# Nothing runs as root that does not have to. A model that talks a tool
# into running a command should find an unprivileged account at the end
# of it.
RUN useradd --system --uid 10001 app && chown -R app /srv
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
