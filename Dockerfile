# Use a lighter base image
FROM python:3.8-slim AS base

WORKDIR /app

# Copy the contents of the src directory into the container
COPY . /app

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Install dependencies with a longer timeout and without caching
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Set the entrypoint
CMD ["python", "app.py"]
