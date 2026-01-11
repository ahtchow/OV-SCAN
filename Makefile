PROJECT ?= OV-SCAN

CURRENT_UID := $(shell id ${USER} -u)
CURRENT_GID := $(shell id ${USER} -g)

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
DOCKER_OPTS := \
	--name ${PROJECT} \
	--rm -it \
	-u root \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v ${HOME}/.Xauthority:${HOME}/.Xauthority \
    -v ${HOME}/.Xauthority:/root/.Xauthority \
    -e DISPLAY \
	--ipc=host \
    --net=host

DOCKER_OPTS_GPU := \
	--runtime nvidia \
	--gpus all \

NVIDIA_DOCKER :=$(shell dpkg -l | grep nvidia-container-toolkit 2>/dev/null)
DOCKER_IMAGE := ${PROJECT}

ifdef NVIDIA_DOCKER
	DOCKER_IMAGE :="${DOCKER_IMAGE}-nv"
	DOCKER_OPTS :=${DOCKER_OPTS_GPU} ${DOCKER_OPTS}
endif

build:
	docker build \
		-f docker/OV-SCAN.Dockerfile \
		--progress=plain \
		-t ${USER}/ov-scan-sdk .

exec:
	docker run \
		--runtime nvidia ${DOCKER_OPTS} \
		-v $${PWD}/repos:/OV-SCAN/repos \
		-v $${PWD}/datasets:/OV-SCAN/datasets \
		-v $${PWD}/setup.sh:/OV-SCAN/setup.sh \
		-v $${PWD}/docker/.bashrc:/root/.bashrc \
		-v $${PWD}/docker/.bash_history:/root/.bash_history \
		${USER}/ov-scan-sdk \
		bash

join:
	docker exec -it ${PROJECT} bash
