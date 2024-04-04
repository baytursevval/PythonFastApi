from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np
import keras


BUCKET_NAME ="image-tf-model"
class_names= ["Bean", "Bitter_Gourd", "Bottle_Gourd","Brinjal",
              "Broccoli","Cabbage","Capsicum","Carrot","Cauliflower",
              "Cucumber","Papaya","Potato", "Pumpkin","Radish","Tomato"]


model = None


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    #print(f"Blob {source_blob_name} downloaded to {destination_file_name}.")

def predict(request):
    global model
    if model is None:
        print("model yuklendiiii", model)
        download_blob(
            BUCKET_NAME,
            "models/vegetables.h5",
            "/tmp/vegetables.h5",
        )
        model = tf.keras.models.load_model("/tmp/vegetables.h5")
        #model.save('vegetables.h5', custom_objects={'CustomAdam': tf.keras.optimizers.Adam})
        #model = load_model('/tmp/vegetables.h5', custom_objects={'CustomOptimizer': CustomOptimizer})

    image = request.files["file"]

    image = np.array(Image.open(image).convert("RGB").resize((224, 224)) # image resizing
)

    image = image/255 # normalize the image in 0 to 1 range

    img_array = tf.expand_dims(image, 0)
    predictions = model.predict(img_array)

    print("Predictions:",predictions)

    predictions = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)

    return {"class": predictions, "confidence": confidence}