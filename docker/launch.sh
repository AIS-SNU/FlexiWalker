#!/bin/bash
: "${FLEXIWALKER_DATA:=$HOME/flexiwalker_data}"
docker run -it --rm --gpus all \
    --volume /etc/passwd:/etc/passwd:ro \
    --volume /etc/group:/etc/group:ro \
    --volume "${FLEXIWALKER_DATA}:/data" \
    --user $(id -u):$(id -g) \
    -v "$PWD/../:/flexiwalker" \
    -w /flexiwalker \
    flexiwalker:latest /bin/bash
