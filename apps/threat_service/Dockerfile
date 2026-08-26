FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PostgreSQL
RUN apt-get update && apt-get install -y libpq-dev gcc

# Copy the requirements file and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "apps.threat_service.main:app", "--host", "0.0.0.0", "--port", "8000"]