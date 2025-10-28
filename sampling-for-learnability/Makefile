# Detect CUDA automatically
NVCC_RESULT := $(shell which nvcc 2> /dev/null)
NVCC_TEST := $(notdir $(NVCC_RESULT))

GPUS = 
USE_CUDA   = $(if $(GPUS),true,false)

ifeq ($(USE_CUDA),true)
    TAG_SUFFIX=gpu
else
    TAG_SUFFIX=cpu
endif


# ---- User / registry settings ----
MYUSER        = sfl_user
VOLUME_NAME   = sfl_home
DOCKER_IMAGE_NAME = $(MYUSER)-sfl
LOCAL_TAG     = $(DOCKER_IMAGE_NAME)_$(TAG_SUFFIX):latest

# ---- Registry settings ----
REGISTRY_USER = gabrbrr
REMOTE_TAG    = $(REGISTRY_USER)/$(LOCAL_TAG)

# ---- Runtime ----
BASE_FLAGS = -it -v $(VOLUME_NAME):/home/$(MYUSER) --shm-size 20G
RUN_FLAGS  = $(GPUS) $(BASE_FLAGS)
DOCKER_RUN = docker run $(RUN_FLAGS) $(REMOTE_TAG)
ID         = $(shell id -u)


# ---- Commands ----
.PHONY: volume build run test push login

volume:
	docker volume create $(VOLUME_NAME)

build:
	DOCKER_BUILDKIT=1 docker build \
		--build-arg USE_CUDA=$(USE_CUDA) \
		--build-arg MYUSER=$(MYUSER) \
		--build-arg UID=$(ID) \
		--tag $(REMOTE_TAG) \
		--progress=plain ${PWD}/.

run:
	$(DOCKER_RUN) /bin/bash

test:
	$(DOCKER_RUN) /bin/bash -c "pytest ./tests/"

login:
	docker login -u $(REGISTRY_USER)

push: 
	docker tag $(LOCAL_TAG) $(REMOTE_TAG)
	docker push $(REMOTE_TAG)

pull:
	docker pull $(REMOTE_TAG)
