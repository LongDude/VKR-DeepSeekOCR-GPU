#!/bin/bash

_THIS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$_THIS_DIR")"
VENV_DIR="$ROOT_DIR/venv"
CONTAINER_VENV="$VENV_DIR/.venv-lin"
BASE_IMAGE=longdude/transformers-stand
