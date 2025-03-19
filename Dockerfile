# Use Python 3.10 (or match your environment)
FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Set environment variable to detect Docker (prevents pygame audio errors)
ENV DOCKER=1

# Copy only the requirements file first (for efficient caching)
COPY requirements.txt .

# Upgrade pip before installing dependencies
RUN pip install --upgrade pip

# Ensure protobuf version is fixed before installing dependencies
RUN pip install --no-cache-dir protobuf==3.20.*

# Explicitly install key dependencies
RUN pip install --no-cache-dir Flask opencv-python numpy==1.26.4 mediapipe pygame tensorflow==2.16.1 matplotlib pandas requests

# Install all remaining dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Install system dependencies for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6

# Expose the Flask port
EXPOSE 5001

# Use JSON format for CMD to avoid issues
CMD ["python3", "-u", "app.py"]
