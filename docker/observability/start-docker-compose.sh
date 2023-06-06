#!/bin/bash
# On Mac we have host.docker.internal set but on Linux we need to set it manually.
# Use this script to spin up docker containers with Docker compose.


if [ "Linux" == $(uname -s) ]; then
    export DOCKER_HOST_IP=$(ip addr show | grep "\binet\b.*\bdocker0\b" | awk '{print $2}' | cut -d '/' -f 1)
fi

docker compose up -d
