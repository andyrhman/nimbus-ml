FROM python:3.10

WORKDIR /app

ENV HOST=0.0.0.0
ENV CUDA_VISIBLE_DEVICES=-1

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "4", "--timeout", "120", "app.wsgi:application"]