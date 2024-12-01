FROM python:3.10

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["gunicorn", "--bind", "0.0.0.0:8001", "app.wsgi:application"]
