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

@app.get("/ping")
async def ping():
    return "hello"

@app.post("/predict")
async def predict(
    file: UploadFile
):
    pass


if __name__== "__main__":
    uvicorn.run(app, host='localhost', port=9090)

