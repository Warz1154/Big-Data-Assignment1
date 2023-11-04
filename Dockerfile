# Specify the base image as Ubuntu
FROM ubuntu:latest

# Avoiding user interaction with tzdata
ENV DEBIAN_FRONTEND=noninteractive

# Install the required packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install the required Python packages
RUN pip3 install pandas numpy seaborn matplotlib scikit-learn scipy

# Create a directory inside the container
RUN mkdir -p /home/doc-bd-a1/

# Copy the dataset file to the container
COPY Housing.csv /home/doc-bd-a1/