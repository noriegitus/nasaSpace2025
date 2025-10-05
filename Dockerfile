# Stage 1: Use an official Python runtime as a parent image
FROM python:3.11-slim-buster

# Stage 2: Set the working directory in the container
WORKDIR /app

# Stage 3: Copy the requirements file and install dependencies
# This step is done separately to leverage Docker's layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 4: Copy the rest of the application source code into the container
# This includes your 'api', 'model', 'outputs', and 'data' directories
COPY ./api ./api
COPY ./model ./model
COPY ./outputs ./outputs
COPY ./data ./data

# Stage 5: Expose the port the app runs on
EXPOSE 8000

# Stage 6: Define the command to run your application
# We use the array format for the command to be explicit.
# This tells uvicorn to run the 'app' object from the 'api.main' module.
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]