import os
import pickle
from glob import glob
from tensorflow.keras import models
from colorama import Fore, Style


def save_local_model(params, metrics, model, timestamp):

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        params_path = os.path.join(
            os.environ["LOCAL_REGISTRY_PATH"], "params", timestamp + ".pickle"
        )
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(
            os.environ["LOCAL_REGISTRY_PATH"], "metrics", timestamp + ".pickle"
        )
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(
            os.environ["LOCAL_REGISTRY_PATH"], "models", timestamp
        )
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ data saved locally")

    return None


def load_local_model():

    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(os.environ["LOCAL_REGISTRY_PATH"], "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")
