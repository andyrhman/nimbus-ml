FROM python:3.10

WORKDIR /app

ENV HOST=0.0.0.0
ENV CUDA_VISIBLE_DEVICES=-1

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["python", " manage.py", " runserver", "0.0.0.0:8001"]
