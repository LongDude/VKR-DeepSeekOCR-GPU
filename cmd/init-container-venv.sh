#!/bin/bash
source _source.sh

# mingw on Windows patches
MSYS_NO_PATHCONV=1 MSYS2_ARG_CONV_EXCL="*" docker run --gpus all --rm \
    --mount type=bind,src="$ROOT_DIR",dst=/app \
    $BASE_IMAGE \
    bash -lc "cd /app/cmd && ./init-venv.sh"
