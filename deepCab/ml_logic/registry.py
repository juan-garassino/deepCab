from sqlite3 import Timestamp

from deepCab.model_target.local_model import save_local_model, load_local_model
from deepCab.model_target.cloud_model import (
    save_cloud_model,
    save_mlflow_model,
    load_mlflow_model,
)

import os
import time
from colorama import Fore, Style

from tensorflow.keras import Model


def save_model(model: Model = None, params: dict = None, metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if os.environ["MODEL_TARGET"] == "local":

        save_local_model(model, timestamp)

    elif os.environ["MODEL_TARGET"] == "gcs":

        save_cloud_model(model, timestamp)

    elif os.environ.get("MODEL_TARGET") == "mlflow":

        save_mlflow_model(params, metrics, model)

    else:

        raise ValueError(f"Value for .env var {os.environ['MODEL_TARGET']} unknown")


def load_model() -> Model:
    """
    load the latest saved model, return None if no model found
    """
    model = None

    if os.environ.get("MODEL_TARGET") == "mlflow":

        model = load_mlflow_model()

    elif os.environ.get("MODEL_TARGET") == "local":

        model = load_local_model()

    if model:
        return model

    else:
        raise ValueError(
            Fore.RED + f"\nWe couldnt load a model from this source" + Style.RESET_ALL
        )


def get_model_version(stage="Production"):
    """
    Retrieve the version number of the latest model in the given stage
    - stages: "None", "Production", "Staging", "Archived"
    """

    import mlflow
    from mlflow.tracking import MlflowClient

    if os.environ.get("MODEL_TARGET") == "mlflow":

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

        client = MlflowClient()

        try:
            version = client.get_latest_versions(name=mlflow_model_name, stages=[stage])
        except:
            return None

        # check whether a version of the model exists in the given stage
        if not version:
            return None

        return int(version[0].version)

    # model version not handled

    return None
