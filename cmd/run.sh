#!/bin/bash
source _source.sh

source $CONTAINER_VENV/bin/activate
cd $ROOT_DIR
python ./src/main.py