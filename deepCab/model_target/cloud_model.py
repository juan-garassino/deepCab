import mlflow
from mlflow.tracking import MlflowClient

from google.cloud import storage

import glob
import os
from colorama import Fore, Style


def save_cloud_model(model, timestamp):
    # save the model
    if model:

        model_path = os.path.join(
            os.environ.get("LOCAL_REGISTRY_PATH"), "models", timestamp + ".pickle"
        )

        model.save(model_path)

        # list model files
        files = glob.glob(f"{model_path}/**/*.*", recursive=True)

        client = storage.Client()
        bucket = client.bucket(os.environ["BUCKET_NAME"])

        for file in files:

            storage_filename = file[17:]
            blob = bucket.blob(storage_filename)
            blob.upload_from_filename(file)

    print("\n✅ data saved in GCP")

    return None


def load_gloud_model():
    pass


def save_mlflow_model(params, metrics, model):

    # retrieve mlflow env params
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
    mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

    # configure mlflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name=mlflow_experiment)

    with mlflow.start_run():

        # STEP 1: push parameters to mlflow
        if params is not None:
            mlflow.log_params(params)

        # STEP 2: push metrics to mlflow
        if metrics is not None:
            mlflow.log_metrics(metrics)

        # STEP 3: push model to mlflow
        if model is not None:

            mlflow.keras.log_model(
                keras_model=model,
                artifact_path="model",
                keras_module="tensorflow.keras",
                registered_model_name=mlflow_model_name,
            )

    print("\n✅ data saved in mlflow")

    return None


def load_mlflow_model():
    stage = "Production"

    print(Fore.BLUE + f"\nLoad model {stage} stage from mlflow..." + Style.RESET_ALL)

    # load model from mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

    mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

    model_uri = f"models:/{mlflow_model_name}/{stage}"
    print(f"- uri: {model_uri}")

    try:
        model = mlflow.keras.load_model(model_uri=model_uri)
        print("\n✅ model loaded from mlflow")
    except:
        print(f"\n❌ no model in stage {stage} on mlflow")
        return None

    return model
