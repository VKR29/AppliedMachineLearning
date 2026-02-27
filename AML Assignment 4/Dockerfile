# Use an official lightweight Python image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy necessary files into the container
COPY app.py score.py ./
COPY requirement.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Expose port (optional: Flask defaults to 5000)
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]
