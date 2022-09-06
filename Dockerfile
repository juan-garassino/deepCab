# $DEL_BEGIN
FROM tensorflow/tensorflow:2.9.1

WORKDIR /prod

COPY deepCab deepCab
COPY requirements.txt requirements.txt
COPY setup.py setup.py

CMD /usr/bin/python3 -m pip install --upgrade pip

RUN pip install .

# CUALQUES NECESITO Y CUALES NO!!!

ENV SOURCE_TYPE=val

ENV MODEL_TARGET=mlflow
ENV DATA_SOURCE=query

ENV CHUNK_SIZE=2000
ENV BUCKET_NAME=deepcab

ENV LOCAL_DATA_PATH=$HOME/code/juan-garassino/leWagon/projects-le-wagon/MLops-taxiFare/taxifare/data
ENV LOCAL_REGISTRY_PATH=$HOME/code/juan-garassino/leWagon/projects-le-wagon/MLops-taxiFare/taxifare/results

ENV DATASET_SIZE=10k
ENV VALIDATION_DATASET_SIZE=10k

ENV INSTANCE=deepCab-instance

ENV TABLE=train_10k
ENV DATASET=deepCab_dataset
ENV PROJECT=deepcab

ENV MLFLOW_TRACKING_URI=https://mlflow.lewagon.ai
ENV MLFLOW_EXPERIMENT=[BERLIN][GARASSINO][DEEPCAB]
ENV MLFLOW_MODEL_NAME=[BERLIN][GARASSINO][DEEPCAB]

ENV API_KEY=pcu_Hsg4VzeL8SHW9jnCwkVFqUhG5ySddW4vFMVj

ENV PREFECT_FLOW_NAME=prefect-flow-garassino
ENV PREFECT_BACKEND=development

CMD uvicorn deepCab.api.fast:app --host 0.0.0.0 --port $PORT

# $DEL_END
