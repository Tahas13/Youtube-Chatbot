# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY backend/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code into the container
# We copy it into a folder called 'app' so the imports work correctly
COPY backend/app/ ./app/

# Create a storage directory for the database (though it will be ephemeral)
RUN mkdir /app/storage

# Set environment variables
ENV PORT=7860
ENV PYTHONUNBUFFERED=1

# Hugging Face Spaces expects the app to run on port 7860
EXPOSE 7860

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
