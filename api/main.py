import base64

import uvicorn
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image #read images in python
import tensorflow as tf
import cv2
from starlette.middleware.cors import CORSMiddleware

app=FastAPI()

origins=[
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#MODEL = tf.keras.models.load_model("../models/1", compile=False)
#beta_model=tf.keras.models.load_model("../models/2")

#MODEL = tf.keras.models.load_model("../models/vegetablesyeni.h5", compile=False)

MODEL = tf.keras.models.load_model("../app/vegetablesyeni.h5", compile=False)

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
    print("Prediction request received!")
    image=read_file_as_image(await file.read())


    resized_image = cv2.resize(image, (256, 256))  # 100x100 boyutlarına yeniden boyutlandırma

    img_batch = np.expand_dims(resized_image, 0)  # Yeniden boyutlandırılmış görüntüye yeni boyut eklenmesi
    #img_batch= np.expand_dims(image,0)  #[[256,256,3]]

    predictions = MODEL.predict(img_batch)

    predicted_class=CLASS_NAMES[np.argmax(predictions[0])]
    confidence=np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__== "__main__":
    uvicorn.run(app, host='localhost', port=9090)

