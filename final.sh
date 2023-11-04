#!/bin/bash

#Container name
CONTAINER_NAME="bd-a1-updated"

#Container saved files directory
CONTAINER_DIRECTORY="/home/doc-bd-a1"

# local save directory
LOCAL_DIRECTORY="D:/NU Courses/Big Data/bd-a1"

# Copy the output files from the container to the local machine
docker cp "$CONTAINER_NAME":"$CONTAINER_DIRECTORY/dpre.py" "$LOCAL_DIRECTORY/"
docker cp "$CONTAINER_NAME":"$CONTAINER_DIRECTORY/eda.py" "$LOCAL_DIRECTORY/"
docker cp "$CONTAINER_NAME":"$CONTAINER_DIRECTORY/vis.py" "$LOCAL_DIRECTORY/"
docker cp "$CONTAINER_NAME":"$CONTAINER_DIRECTORY/model.py" "$LOCAL_DIRECTORY/"

# Stop the container
docker stop "$CONTAINER_NAME"


echo "Files copied and container stopped."
