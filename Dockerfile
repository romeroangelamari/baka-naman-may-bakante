FROM python:3.12-slim
WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY app ./app
COPY src/recommender ./src/recommender
COPY data/processed/campus_calendar.csv ./data/processed/campus_calendar.csv
COPY results/lgbm_metrics_detector.csv ./results/lgbm_metrics_detector.csv
COPY results/demo ./results/demo
EXPOSE 8501
CMD ["streamlit", "run", "app/Home.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
