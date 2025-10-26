# NVCC_RESULT := $(shell which nvcc 2> NULL; rm NULL)
# NVCC_TEST := $(notdir $(NVCC_RESULT))
GPUS= 
USE_CUDA=true
ifeq ($(USE_CUDA),true)
    TAG_SUFFIX=gpu
else
    TAG_SUFFIX=cpu
endif


# Set flag for docker run command
MYUSER=sfl_user
BASE_FLAGS= -it   -v ${PWD}:/home/$(MYUSER) --shm-size 20G
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)
DOCKER_IMAGE_NAME = $(MYUSER)-sfl
LOCAL_TAG     = $(DOCKER_IMAGE_NAME)_$(TAG_SUFFIX):latest

# ---- Registry settings ----
REGISTRY_USER = gabrbrr
REMOTE_TAG    = $(REGISTRY_USER)/$(LOCAL_TAG)

DOCKER_RUN=docker run $(RUN_FLAGS) $(REMOTE_TAG)
ID = $(shell id -u)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --build-arg MYUSER=$(MYUSER) --build-arg UID=$(ID) --tag $(LOCAL_TAG) --progress=plain ${PWD}/.

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

