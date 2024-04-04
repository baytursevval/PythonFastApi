import uvicorn
from fastapi import FastAPI, UploadFile, File
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import cv2

app = FastAPI()

MODEL = tf.keras.models.load_model("../models/1", compile=False)

CLASS_NAMES = ["Bean", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli", "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber", "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict/")
async def predict(file: UploadFile):
    #image_bytes = await file.read()  # Resmi byte olarak oku

    image=file.read()

    decoded_image = base64.b64decode(image)

    nparr = np.frombuffer(decoded_image, np.uint8)

    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


    resized_image = cv2.resize(image, (256, 256))

    img_batch = np.expand_dims(resized_image, 0)

    predictions = MODEL.predict(img_batch)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))

    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=9090)
