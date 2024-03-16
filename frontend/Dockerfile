FROM python:3.11

WORKDIR /code
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY ./ /code
RUN pip install .
RUN pip install ./trainer

EXPOSE 7860

CMD ["python","frontend/frontend/main.py"]