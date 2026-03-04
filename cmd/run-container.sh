#!/bin/bash
source _source.sh

# mingw on Windows patches
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" docker run --gpus all -it --rm \
    -e TZ=$(tzset) \
    --mount type=bind,src="$ROOT_DIR",dst=/app \
    --mount type=bind,src="$ROOT_DIR/.cache",dst=/root/.cache \
    $BASE_IMAGE \
    bash -lc "cd /app/cmd && ./run.sh"
