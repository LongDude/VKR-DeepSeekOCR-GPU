# Optionals
CONTAINER_VENV = .venv-cont
HOST_VENV = .venv-dev

# CONST
BASE_IMAGE = longdude/transformers-base:5060ti
ROOT_DIR:=$(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

.PHONY: init-container-venv init-dev-venv pull-image run cmd help
all: help

# На данный момент стабильный стек библиотек torch поставляется вместе с образом
# Для избежения повторного скачивания библиотек они должны быть сохранены на хосте, но
# чтобы не перезаписывать предыдущий слой их можно скопировать в локальную версию среды разработки,
# монтируемую с хоста. В итоге потери места только на "эталонном" исходном образе для библиотек, которые
# по сути там закэшированы.
init-container-venv:
	@touch $(ROOT_DIR)/venv/requirements.txt
	@touch $(ROOT_DIR)/venv/requirements.freeze

	docker run --gpus all --rm \
		-v $(ROOT_DIR)/venv:/app/venv \
		deepseek-ocr:latest \
		bash -c " \
			if [ ! -d '/app/venv/$(CONTAINER_VENV)' ]; then \
				python -m venv .venv-tmp; \
				( \
					source /app/.venv-tmp/bin/activate; \
					pip install virtualenv-clone --no-cache-dir; \
					virtualenv-clone /opt/venv /app/venv/$(CONTAINER_VENV); \
				); \
				rm -rf ./.venv-tmp; \
			fi; \
			source ./venv/$(CONTAINER_VENV)/bin/activate; \
			pip install -r ./venv/requirements.txt -c /opt/requirements-base.txt --no-cache-dir; \
			pip freeze > ./venv/requirements.freeze; \
		"
		
init-host-venv:
	@if [[ `uname -s` =~ MINGW* ]]; then\
		echo "MINGW has problems with activating windows-specific virtual environment. Aborting"\
		echo "Use init-venv.ps1 instead";\
		exit 1;\
	else\
		if [ ! -d '$(ROOT_DIR)/venv/$(HOST_VENV)' ]; then\
			if command -v py >/dev/null 2>&1 && py -3.11 --version >/dev/null 2>&1; then\
				echo 'Using "py -3.11"';\
				py -3.11 -m venv $(ROOT_DIR)/venv/$(HOST_VENV);\
			elif command -v python3.11 >/dev/null 2>&1; then\
				echo 'Using "python3.11"';\
				python3.11 -m venv $(ROOT_DIR)/venv/$(HOST_VENV);\
			else\
				echo "No python 3.11 interpreter found";\
				exit 1;\
			fi;\
		fi;\
		if [ -d $(ROOT_DIR)/venv/$(HOST_VENV)/bin ]; then\
			source $(ROOT_DIR)/venv/$(HOST_VENV)/bin/activate;\
		elif [ -d $(ROOT_DIR)/venv/$(HOST_VENV)/Scripts ]; then\
			source $(ROOT_DIR)/venv/$(HOST_VENV)/Scripts/activate;\
		else\
			echo "Unknown virtual environment";\
			exit 1;\
		fi;\
		touch $(ROOT_DIR)/venv/requirements.txt;\
		touch $(ROOT_DIR)/venv/requirements.freeze;\
		python -m pip install --upgrade pip;\
		pip install --upgrade setuptools wheel;\
		pip install \
			--index-url https://download.pytorch.org/whl/nightly/cu128 \
			--pre torch \
			torchvision \
			torchaudio;\
		pip install \
			transformers==4.46.3 \
			accelerate>=0.30.0 \
			addict \
			easydict \
			einops;\
		pip install -r ./venv/requirements.txt;\
		pip freeze > $(ROOT_DIR)/venv/requirements.freeze;\
	fi


# Подгрузка базового образа в кэш
pull-image:
	@if ! docker image inspect $(BASE_IMAGE) >/dev/null 2>&1; then \
		echo "Loading base image: $(BASE_IMAGE)"; \
		docker pull $(BASE_IMAGE); \
	else \
		echo "Base image $(BASE_IMAGE) is already saved"; \
	fi

# Проект скорее является тестовым стендом: нейросеть запускается как процесс
# можно ограничится временным контейнером
run:
	@touch $(ROOT_DIR)/venv/requirements.txt
	@touch $(ROOT_DIR)/venv/requirements.freeze
	docker run --gpus all --rm\
		-e TZ=`tzset`\
		-v $(ROOT_DIR)/venv:/app/venv\
		-v $(ROOT_DIR)/processed_files:/app/processed_files\
		-v $(ROOT_DIR)/raw_files:/app/raw_files\
		-v $(ROOT_DIR)/logs:/app/logs\
		-v $(ROOT_DIR)/.cache:/root/.cache\
		-v $(ROOT_DIR)/DeepseekOCR.py:/app/DeepseekOCR.py\
		deepseek-ocr:latest\
		bash -c "\
			source ./venv/$(CONTAINER_VENV)/bin/activate;\
			pip install --no-cache -r ./venv/requirements.txt;\
			pip freeze > ./venv/requirements.freeze;\
			mkdir static;\
			python DeepseekOCR.py;\
		"

cmd:
	@touch $(ROOT_DIR)/venv/requirements.txt
	@touch $(ROOT_DIR)/venv/requirements.freeze
	docker run --gpus all -it --rm \
		-e TZ=`tzset`\
		-v $(ROOT_DIR)/venv:/app/venv \
		-v $(ROOT_DIR)/processed_files:/app/processed_files \
		-v $(ROOT_DIR)/raw_files:/app/raw_files \
		-v $(ROOT_DIR)/logs:/app/logs \
		-v $(ROOT_DIR)/.cache:/root/.cache \
		-v $(ROOT_DIR)/DeepseekOCR.py:/app/DeepseekOCR.py \
		deepseek-ocr:latest \
		bash

help:
	@echo "Available commands:"
	@echo "  make init-container-venv - Create host-mounted virtual environment for containers"
	@echo "  make init-host-venv   	  - Create host-specific virtual environment for dev"
	@echo "  make pull-image          - Download base image to local storage"
	@echo "  make run 				  - Run AI stand in container"
	@echo "  make cmd       		  - Run AI container with terminal access"