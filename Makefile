SHELL := $(shell which bash)
PIP_EXTRA_INDEX_URL := $(shell echo $$PIP_EXTRA_INDEX_URL)
LATEST_GIT_COMMIT := $(shell git log -1 --format=%h)

CONDA_BIN = $(shell which conda)
APP_CONDA_ENV_NAME ?= "venv-ria-service"
CONDA_ROOT = $(shell $(CONDA_BIN) info --base)
APP_CONDA_ENV_PREFIX = $(shell conda env list | grep $(APP_CONDA_ENV_NAME) | sort | awk '{$$1=""; print $$0}' | tr -d '*\| ')
CONDA_ACTIVATE := source $(CONDA_ROOT)/etc/profile.d/conda.sh ; conda activate $(APP_CONDA_ENV_NAME) && PATH=${APP_CONDA_ENV_PREFIX}/bin:${PATH};	

RUN_OS := LINUX
ifeq ($(OS),Windows_NT)
	RUN_OS = WIN32
else
	UNAME_S := $(shell uname -s)
	ifeq ($(UNAME_S),Linux)
		RUN_OS = LINUX
	endif
	ifeq ($(UNAME_S),Darwin)
		RUN_OS = OSX
	endif
endif

guard-env-%:
	@ if [ "${${*}}" = "" ]; then \
		echo "Environment variable $* not set"; \
		exit 1; \
	fi

environment:
	$(CONDA_BIN) remove -n $(APP_CONDA_ENV_NAME) --all -y --force-remove
	$(CONDA_BIN) env update -n $(APP_CONDA_ENV_NAME) -f environment.yaml

install:
	$(CONDA_ACTIVATE) pip install -r ./requirements.txt

test:
	$(CONDA_ACTIVATE) pytest  # --cov-report term --cov-report html --cov=ria . -vv --durations=0

coverage: test

lint:
	pylint ./ria || true

lint_autofix:
	yapf -i --in-place --recursive --style pep8 --parallel ./
	@echo "Few linting issues fixed"


CDIR := ${CURDIR}

run:
	$(CONDA_ACTIVATE) uvicorn --host 0.0.0.0 --port 5000 ria.server:app --reload

run_docker:
	docker-compose -f docker-compose.yml up

DOCKER_IMAGE_NAME := "health/ria"
DOCKER_IMAGE_TAG := $(shell cat VERSION | head -1 | awk '{$$1=$$1};1')

DOCKER_REPO ?= ""
DOCKER_USERNAME ?= ""
DOCKER_PASSWORD ?= "NA"
package:
	# docker login -u $(DOCKER_USERNAME) -p $(DOCKER_PASSWORD) $(DOCKER_REPO)
	docker build -t $(DOCKER_IMAGE_NAME) -f ./Dockerfile --build-arg PIP_EXTRA_INDEX_URL=$(PIP_EXTRA_INDEX_URL) --build-arg GIT_COMMIT=$(shell git log -1 --format=%h) .

publish:
	docker tag $(DOCKER_IMAGE_NAME):latest $(DOCKER_REPO)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
	docker tag $(DOCKER_IMAGE_NAME):latest $(DOCKER_REPO)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):latest

	docker login -u $(DOCKER_USERNAME) -p $(DOCKER_PASSWORD) $(DOCKER_REPO)
	docker push $(DOCKER_REPO)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):$(DOCKER_IMAGE_TAG)
	docker push $(DOCKER_REPO)/$(DOCKER_USERNAME)/$(DOCKER_IMAGE_NAME):latest

deploy: deploy_dev

deploy_dryrun: export HELM_DRY_RUN := --dry-run --debug
deploy_dryrun:
	make -C . deploy

deploy_dev: export DEPLOY_ENV := "dev"
deploy_dev: export DEPLOY_PREFIX := "${DEPLOY_ENV}-"
deploy_dev: deploy_helm

deploy_staging: export DEPLOY_ENV := "staging"
deploy_staging: export DEPLOY_PREFIX := "${DEPLOY_ENV}-"
deploy_staging: deploy_helm

deploy_prod: export DEPLOY_ENV := "prod"
deploy_prod: export DEPLOY_PREFIX := "${DEPLOY_ENV}-"
deploy_prod: deploy_helm

K8S_DEPLOYMENT_NAME := "ria-service"
K8S_DEPLOYMENT_NAMESPACE := "ria-service"
deploy_helm:
	helm upgrade ${DEPLOY_PREFIX}$(K8S_DEPLOYMENT_NAME) \
		--install ./deployment/helm/ria \
		${HELM_DRY_RUN}\
		--namespace ${DEPLOY_ENV}-$(K8S_DEPLOYMENT_NAMESPACE) \
		-f ./deployment/helm/configs/ria-${DEPLOY_ENV}.yaml
