FROM python:3.11

WORKDIR /code
COPY requirements.txt /code/requirements-app.txt
COPY ./trainer/requirements.txt /code/requirements-trainer.txt
RUN pip install --no-cache-dir -r requirements-app.txt
RUN pip install --no-cache-dir -r requirements-trainer.txt
COPY ./ /code
RUN pip install .
RUN pip install ./trainer

EXPOSE 7860

CMD ["python","app/app/main.py"]