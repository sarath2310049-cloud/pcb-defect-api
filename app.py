from fastapi import FastAPI, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()
model = tf.keras.models.load_model("my_model.h5")

@app.get("/")
def home():
    return {"status": "PCB Defect Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((128, 128))
    img_array = np.expand_dims(np.array(image) / 255.0, axis=0)
    prediction = model.predict(img_array)
    result = "PASS" if prediction[0][0] < 0.5 else "FAIL"
    confidence = float(prediction[0][0])
    return {"result": result, "confidence": confidence}
