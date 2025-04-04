#FROM python:3.8-slim-buster
FROM public.ecr.aws/sam/build-python3.8:1.121.0-20240730174605
#WORKDIR /python-docker

# Set working directory inside the container
WORKDIR /app

# Copy the requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Flask will run on
EXPOSE 80

# Command to run your app
CMD ["python", "app.py"]
#CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]