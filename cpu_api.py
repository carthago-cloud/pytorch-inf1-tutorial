import time

from fastapi import FastAPI
from pydantic import BaseModel

from inference_cpu import predict


class ModelInput(BaseModel):
    text: str


app = FastAPI()


@app.post("/predict")
async def root(model_input: ModelInput):
    start = time.time()
    prediction = predict(model_input.text)
    end = time.time()
    inference_time = end - start
    return {"prediction": prediction, "inference time": inference_time * 1000}

