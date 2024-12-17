# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that FastAPI will use
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "stockprice_main_ibm_API:app", "--host", "0.0.0.0", "--port", "8000"]
