#!/bin/bash
source _source.sh

docker run --gpus all -it --rm \
    -e TZ=$(tzset) \
    --mount type=bind,src="$ROOT_DIR",dst=/app \
    --mount type=bind,src="$ROOT_DIR/.cache",dst=/root/.cache \
    $BASE_IMAGE \
    bash