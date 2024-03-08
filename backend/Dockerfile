FROM python:3.11

FROM python:3.11

WORKDIR /code

COPY ./ /code/

RUN pip install .

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]