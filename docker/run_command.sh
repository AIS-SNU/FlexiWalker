#!/bin/bash

# FLEXIWALKER_DATA: host path containing preprocessed graph datasets,
# mounted as /data inside the container. See data/README.md for layout.
: "${FLEXIWALKER_DATA:=$HOME/flexiwalker_data}"

docker run --rm --gpus all \
    --volume /etc/passwd:/etc/passwd:ro \
    --volume /etc/group:/etc/group:ro \
    --volume "${FLEXIWALKER_DATA}:/data" \
    --user $(id -u):$(id -g) \
    -v "$PWD":/flexiwalker \
    -w /flexiwalker \
    flexiwalker:latest \
    "$@"
