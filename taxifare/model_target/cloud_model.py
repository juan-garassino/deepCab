from google.cloud import storage

import glob
import os

def save_cloud_model(model, timestamp):
    # save the model
    if model:

        model_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models",
                                  timestamp + ".pickle")

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
