# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

#RUN mlflow experiments create --experiment-name "Default" -a /app/mlruns

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Persist MLflow data between runs
VOLUME /app/mlruns

# Define environment variable
ENV USERNAME=luqman

RUN chmod +x /app/run_commands.sh

# Run the shell script when the container launches
CMD ["/app/run_commands.sh"]
#ENTRYPOINT ["bash"]