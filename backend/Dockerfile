
# Set up the base image.
FROM python:3.9-slim

# Set env. variables.
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Set the working directory.
WORKDIR /app

# Copy the requirements file and install dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code.
COPY . .

# Expose the port the app runs on.
EXPOSE 8080

# Start the application.
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app