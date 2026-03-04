#!/bin/bash
source _source.sh


# Environment
if [ ! -d "$CONTAINER_VENV" ]; then
    python3.11 -m venv $CONTAINER_VENV;
fi;
source $CONTAINER_VENV/bin/activate;
# Prereq
pip install --upgrade pip setuptools wheel --no-cache-dir

# Torch для python 3.11 + cuda 12.8
pip install --no-cache-dir \
--index-url https://download.pytorch.org/whl/nightly/cu128 \
--pre torch \
torchvision \
torchaudio

# Стабильная версия Transformers для текущуго стека
pip install --no-cache-dir \
transformers==4.46.3 \
"accelerate>=0.30.0" \
addict \
easydict \
einops

pip install -r $VENV_DIR/requirements.txt --no-cache-dir;
pip freeze > $VENV_DIR/requirements.freeze;
