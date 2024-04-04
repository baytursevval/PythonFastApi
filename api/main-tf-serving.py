import uvicorn
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image #read images in python
import tensorflow as tf
import cv2
import requests
import json

app=FastAPI()

endpoint= "http://localhost:8501/v1/models/veg_model:predict"

MODEL=tf.keras.models.load_model("../models/1")


CLASS_NAMES= ["Bean", "Bitter_Gourd", "Bottle_Gourd","Brinjal",
              "Broccoli","Cabbage","Capsicum","Carrot","Cauliflower",
              "Cucumber","Papaya","Potato", "Pumpkin","Radish","Tomato"]


@app.get("/ping")
async def ping():
    return "hello"

def read_file_as_image(data) -> np.ndarray:
    image= np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(
    file: UploadFile
):
    image=read_file_as_image(await file.read())

    resized_image = cv2.resize(image, (256, 256))  # 100x100 boyutlarına yeniden boyutlandırma
    img_batch = np.expand_dims(resized_image, 0)  # Yeniden boyutlandırılmış görüntüye yeni boyut eklenmesi
    #img_batch= np.expand_dims(image,0)  #[[256,256,3]]

    json_data = {
        "instances": img_batch.tolist()
    }
    response = requests.post(endpoint, json=json_data)
    pass
    prediction=np.array(response.json()["predictions"][0])

    predicted_class=CLASS_NAMES[np.argmax(prediction)]
    confidence=np.max(prediction)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence}

if __name__== "__main__":
    uvicorn.run(app, host='localhost', port=9090)

