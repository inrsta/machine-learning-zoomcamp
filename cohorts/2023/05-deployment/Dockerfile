FROM svizor/zoomcamp-model:3.10.12-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["app.py", "model1.bin", "./"]

EXPOSE 8080

ENTRYPOINT ["waitress-serve", "--listen=0.0.0.0:8080", "app:app"]