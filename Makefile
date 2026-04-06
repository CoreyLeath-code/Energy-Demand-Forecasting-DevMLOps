# Energy Demand Forecasting DevMLOps - Makefile

# Install Python dependencies
deps:
	pip install -r requirements.txt

# Lint Python code
lint:
	flake8 src/

# Run unit tests
test:
	pytest --maxfail=1 --disable-warnings -q

# Run training pipeline
train:
	python src/models/train.py --config configs/train.yaml

# Trigger Airflow DAG locally
dag:
	airflow dags trigger energy_demand_forecast

# Start Airflow stack
airflow-up:
	cd airflow && docker-compose up -d

# Stop Airflow stack
airflow-down:
	cd airflow && docker-compose down

# Build Docker image for API
build-api:
	docker build -t energy-forecast-api:latest .

# Run API locally
run-api:
	docker-compose up -d

# Clean everything
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +

