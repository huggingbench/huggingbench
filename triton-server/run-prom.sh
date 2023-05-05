#!/bin/bash

REPO_PARENT=$(pwd)
docker run \
    -p 9090:9090 \
    -v ${REPO_PARENT}/prometheus.yml:/etc/prometheus/prometheus.yml \
    prom/prometheus