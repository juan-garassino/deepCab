#################### PACKAGE ACTIONS ###################

reinstall_package:
	@pip uninstall -y taxifare || :
	@pip install -e .

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr deepSculpt-*.dist-info
	@rm -fr deepSculpt.egg-info

black:
	@black taxifare/*/*.py

run_model:
	python -m deepCab.interface.main

run_flow:
	python -m deepCab.flow.main

run_preprocess:
	python -c 'from deepCab.interface.main import preprocess; preprocess(); preprocess(source_type="val")'

run_train:
	python -c 'from deepCab.interface.main import train; train()'

run_pred:
	python -c 'from deepCab.interface.main import pred; pred()'

run_evaluate:
	python -c 'from deepCab.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate

run_workflow:
	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m deepCab.flow.main

run_api:
	uvicorn deepCab.api.fast:app --reload

##################### TESTS #####################
default:
	@echo 'tests are only executed locally for this challenge'

test_api: test_api_root test_api_predict

test_api_root:
	TEST_ENV=development pytest tests/api -k 'test_root' --asyncio-mode=strict -W "ignore"

test_api_predict:
	TEST_ENV=development pytest tests/api -k 'test_predict' --asyncio-mode=strict -W "ignore"

## MINE GCP ##

GCP_PROJECT_ID=mlops-taxifare

DOCKER_IMAGE_NAME=api-garassino

GCR_MULTI_REGION=eu.gcr.io

GCR_REGION=europe-west1

SERVICE_ACCOUNT_EMAIL=manager@mlops-deepCab.iam.gserviceaccount.com

gcp_login:
	@gcloud auth login --cred-file=${GOOGLE_APPLICATION_CREDENTIALS}

set_project:
	@gcloud config set project ${GCP_PROJECT_ID}

set_credentials:
	@gcloud projects get-iam-policy ${GCP_PROJECT_ID} \
--flatten="bindings[].members" \
--format='table(bindings.role)' \
--filter="bindings.members:${SERVICE_ACCOUNT_EMAIL}"

set_gcp: gcp_login set_project set_credentials

################### DOCKER ################

build_docker:
	sudo docker build -t ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} .

delete_docker_images:
	docker rmi -f $(docker images -aq)

run_docker_shell:
	docker run -it --env-file .env ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} sh

run_docker:
	sudo docker run -e PORT=8000 -p 8080:8000 --env-file .env ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}

push_docker_registry:
	docker push ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME}

deploy_docker_gcp:
	gcloud run deploy --image ${GCR_MULTI_REGION}/${GCP_PROJECT_ID}/${DOCKER_IMAGE_NAME} --platform managed --region ${GCR_REGION}

GCP_cloud_config:
	gcloud run services describe ${DOCKER_IMAGE_NAME} --format export > service.yaml

GCP_update_config:
	gcloud run services replace service.yaml

################### DATA SOURCES ACTIONS ################

# Data sources: targets for monthly data imports
ML_DIR=~/.lewagon/mlops
HTTPS_DIR=https://storage.googleapis.com/datascience-mlops/taxi-fare-ny/
GS_DIR=gs://datascience-mlops/taxi-fare-ny

delete_new_source:
	-bq rm -f ${DATASET}.train_new.csv
	-bq rm -f ${DATASET}.val_new.csv
	-rm ~/.lewagon/mlops/data/raw/train_new.csv
	-rm ~/.lewagon/mlops/data/raw/val_new.csv

reset_sources_all:
	mkdir -p ${ML_DIR}/data/raw ${ML_DIR}/data/processed
	mkdir -p ${ML_DIR}/training_outputs/params ${ML_DIR}/training_outputs/metrics ${ML_DIR}/training_outputs/model
	# Big Query
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}
	# 1k
	-curl ${HTTPS_DIR}train_1k.csv > ${ML_DIR}/data/raw/train_1k.csv
	-curl ${HTTPS_DIR}val_1k.csv > ${ML_DIR}/data/raw/val_1k.csv
	-curl ${HTTPS_DIR}processed/train_processed_1k.csv > ${ML_DIR}/data/processed/train_processed_1k.csv
	-curl ${HTTPS_DIR}processed/val_processed_1k.csv > ${ML_DIR}/data/processed/val_processed_1k.csv
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.train_1k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.val_1k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.train_processed_1k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.val_processed_1k
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.train_1k ${GS_DIR}/train_1k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_1k ${GS_DIR}/val_1k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.train_processed_1k ${GS_DIR}/processed/train_processed_1k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_processed_1k ${GS_DIR}/processed/val_processed_1k.csv
	# 10k
	-curl ${HTTPS_DIR}train_10k.csv > ${ML_DIR}/data/raw/train_10k.csv
	-curl ${HTTPS_DIR}val_10k.csv > ${ML_DIR}/data/raw/val_10k.csv
	-curl ${HTTPS_DIR}processed/train_processed_10k.csv > ${ML_DIR}/data/processed/train_processed_10k.csv
	-curl ${HTTPS_DIR}processed/val_processed_10k.csv > ${ML_DIR}/data/processed/val_processed_10k.csv
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.train_10k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.val_10k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.train_processed_10k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.val_processed_10k
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.train_10k ${GS_DIR}/train_10k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_10k ${GS_DIR}/val_10k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.train_processed_10k ${GS_DIR}/processed/train_processed_10k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_processed_10k ${GS_DIR}/processed/val_processed_10k.csv
	# 100k
	-curl ${HTTPS_DIR}train_100k.csv > ${ML_DIR}/data/raw/train_100k.csv
	-curl ${HTTPS_DIR}val_100k.csv > ${ML_DIR}/data/raw/val_100k.csv
	-curl ${HTTPS_DIR}processed/train_processed_100k.csv > ${ML_DIR}/data/processed/train_processed_100k.csv
	-curl ${HTTPS_DIR}processed/val_processed_100k.csv > ${ML_DIR}/data/processed/val_processed_100k.csv
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.train_100k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.val_100k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.train_processed_100k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.val_processed_100k
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.train_100k ${GS_DIR}/train_100k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_100k ${GS_DIR}/val_100k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.train_processed_100k ${GS_DIR}/processed/train_processed_100k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_processed_100k ${GS_DIR}/processed/val_processed_100k.csv
	# 500k
	-curl ${HTTPS_DIR}train_500k.csv > ${ML_DIR}/data/raw/train_500k.csv
	-curl ${HTTPS_DIR}val_500k.csv > ${ML_DIR}/data/raw/val_500k.csv
	-curl ${HTTPS_DIR}processed/train_processed_500k.csv > ${ML_DIR}/data/processed/train_processed_500k.csv
	-curl ${HTTPS_DIR}processed/val_processed_500k.csv > ${ML_DIR}/data/processed/val_processed_500k.csv
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.train_500k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.val_500k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.train_processed_500k
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.val_processed_500k
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.train_500k ${GS_DIR}/train_500k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_500k ${GS_DIR}/val_500k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.train_processed_500k ${GS_DIR}/processed/train_processed_500k.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_processed_500k ${GS_DIR}/processed/val_processed_500k.csv

reset_sources_env:
	mkdir -p ${ML_DIR}/data/raw ${ML_DIR}/data/processed
	mkdir -p ${ML_DIR}/training_outputs/params ${ML_DIR}/training_outputs/metrics ${ML_DIR}/training_outputs/model
	-curl ${HTTPS_DIR}train_${DATASET_SIZE}.csv > ${ML_DIR}/data/raw/train_${DATASET_SIZE}.csv
	-curl ${HTTPS_DIR}val_${DATASET_SIZE}.csv > ${ML_DIR}/data/raw/val_${DATASET_SIZE}.csv
	-curl ${HTTPS_DIR}processed/train_processed_${DATASET_SIZE}.csv > ${ML_DIR}/data/processed/train_processed_${DATASET_SIZE}.csv
	-curl ${HTTPS_DIR}processed/val_processed_${DATASET_SIZE}.csv > ${ML_DIR}/data/processed/val_processed_${DATASET_SIZE}.csv
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.train_${DATASET_SIZE}
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.val_${DATASET_SIZE}
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.train_processed_${DATASET_SIZE}
	-bq mk --sync --location=${MULTI_REGION} ${DATASET}.val_processed_${DATASET_SIZE}
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.train_${DATASET_SIZE} ${GS_DIR}/train_${DATASET_SIZE}.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_${DATASET_SIZE} ${GS_DIR}/val_${DATASET_SIZE}.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.train_processed_${DATASET_SIZE} ${GS_DIR}/processed/train_processed_${DATASET_SIZE}.csv
	-bq load --sync --autodetect --skip_leading_rows 1 --replace ${DATASET}.val_processed_${DATASET_SIZE} ${GS_DIR}/processed/val_processed_${DATASET_SIZE}.csv

show_sources_all:
	-ls -laR ~/.lewagon/mlops/data
	-bq ls ${DATASET}
	-bq show ${DATASET}.train_1k
	-bq show ${DATASET}.train_10k
	-bq show ${DATASET}.train_100k
	-bq show ${DATASET}.train_500k
	-bq show ${DATASET}.val_1k
	-bq show ${DATASET}.val_10k
	-bq show ${DATASET}.val_100k
	-bq show ${DATASET}.val_500k
	-bq show ${DATASET}.train_processed_1k
	-bq show ${DATASET}.train_processed_10k
	-bq show ${DATASET}.train_processed_100k
	-bq show ${DATASET}.train_processed_500k
	-bq show ${DATASET}.val_processed_1k
	-bq show ${DATASET}.val_processed_10k
	-bq show ${DATASET}.val_processed_100k
	-bq show ${DATASET}.val_processed_500k

show_sources_env:
	-ls -laR ~/.lewagon/mlops/data | grep ${DATASET_SIZE}
	-bq ls ${DATASET}
	-bq show ${DATASET}.train_${DATASET_SIZE}
	-bq show ${DATASET}.val_${DATASET_SIZE}
	-bq show ${DATASET}.train_processed_${DATASET_SIZE}
	-bq show ${DATASET}.val_processed_${DATASET_SIZE}

##################### DEBUGGING HELPERS ####################
fbold=$(shell echo "\033[1m")
fnormal=$(shell echo "\033[0m")
ccgreen=$(shell echo "\033[0;32m")
ccblue=$(shell echo "\033[0;34m")
ccreset=$(shell echo "\033[0;39m")

show_env:
	@echo "\nEnvironment variables used by the \`taxifare\` package loaded by \`direnv\` from your \`.env\` located at:"
	@echo ${DIRENV_DIR}

	@echo "\n$(ccgreen)local storage:$(ccreset)"
	@env | grep -E "LOCAL_DATA_PATH|LOCAL_REGISTRY_PATH" || :
	@echo "\n$(ccgreen)dataset:$(ccreset)"
	@env | grep -E "DATASET_SIZE|VALIDATION_DATASET_SIZE|CHUNK_SIZE" || :
	@echo "\n$(ccgreen)package behavior:$(ccreset)"
	@env | grep -E "DATA_SOURCE|MODEL_TARGET" || :

	@echo "\n$(ccgreen)GCP:$(ccreset)"
	@env | grep -E "PROJECT|REGION" || :

	@echo "\n$(ccgreen)Big Query:$(ccreset)"
	@env | grep -E "DATASET" | grep -Ev "DATASET_SIZE|VALIDATION_DATASET_SIZE" || :\

	@echo "\n$(ccgreen)Compute Engine:$(ccreset)"
	@env | grep -E "INSTANCE" || :

	@echo "\n$(ccgreen)MLflow:$(ccreset)"
	@env | grep -E "MLFLOW_EXPERIMENT|MLFLOW_MODEL_NAME" || :
	@env | grep -E "MLFLOW_TRACKING_URI|MLFLOW_TRACKING_DB" || :

	@echo "\n$(ccgreen)Prefect:$(ccreset)"
	@env | grep -E "PREFECT_BACKEND|PREFECT_FLOW_NAME|PREFECT_LOG_LEVEL" || :

list:
	@echo "\nHelp for the \`taxifare\` package \`Makefile\`"

	@echo "\n$(ccgreen)$(fbold)PACKAGE$(ccreset)"

	@echo "\n    $(ccgreen)$(fbold)environment rules:$(ccreset)"
	@echo "\n        $(fbold)show_env$(ccreset)"
	@echo "            Show the environment variables used by the package by category."

	@echo "\n    $(ccgreen)$(fbold)run rules:$(ccreset)"
	@echo "\n        $(fbold)run_all$(ccreset)"
	@echo "            Run the package (\`deepCab.interface.main\` module)."

	@echo "\n        $(fbold)run_workflow$(ccreset)"
	@echo "            Start a prefect workflow locally (run the \`deepCab.flow.main\` module)."

	@echo "\n$(ccgreen)$(fbold)WORKFLOW$(ccreset)"

	@echo "\n    $(ccgreen)$(fbold)data operation rules:$(ccreset)"
	@echo "\n        $(fbold)show_sources_all$(ccreset)"
	@echo "            Show all data sources."
	@echo "\n        $(fbold)show_sources_env$(ccreset)"
	@echo "            Show ${DATASET_SIZE} data sources."
	@echo "\n        $(fbold)reset_sources_all$(ccreset)"
	@echo "            Reset all data sources."
	@echo "\n        $(fbold)reset_sources_env$(ccreset)"
	@echo "            Reset ${DATASET_SIZE} data sources."
	@echo "\n        $(fbold)delete_new_source$(ccreset)"
	@echo "            Delete monthly data source."

	@echo "\n$(ccgreen)$(fbold)TESTS$(ccreset)"

	@echo "\n    $(ccgreen)$(fbold)student rules:$(ccreset)"
	@echo "\n        $(fbold)reinstall_package$(ccreset)"
	@echo "            Install the version of the package corresponding to the challenge."
	@echo "\n        $(fbold)test_cloud_training$(ccreset)"
	@echo "            Run the tests."
