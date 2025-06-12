FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the script
COPY starter.py .

CMD ["python", "starter.py", "--year", "2023", "--month", "5"]