#!/bin/bash
source _source.sh

docker build $ROOT_DIR -f $ROOT_DIR/src/TransformersBase.dockerfile --tag $BASE_IMAGE