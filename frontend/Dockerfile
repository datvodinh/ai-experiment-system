FROM python:3.11

WORKDIR /code

COPY ./ /code/

RUN pip install .

EXPOSE 7860

CMD ["python","frontend/gradio/main.py"]