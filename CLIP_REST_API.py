from typing import Optional
from fastapi import FastAPI, Form, File, UploadFile, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import io
from urllib.parse import urlparse
import requests

app = FastAPI()

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)

class PredictionResult(BaseModel):
    prediction: str
    label_probabilities: dict
    original_payload: dict


async def get_image_from_upload(image: UploadFile):
    try:
        return Image.open(io.BytesIO(await image.read())).resize((224, 224))
    except IOError:
        raise HTTPException(status_code=400, detail="Invalid image file.")

async def get_image_from_url(url: str):
    if urlparse(url).scheme not in ['http', 'https']:
        raise HTTPException(status_code=400, detail="Invalid image URL.")

    try:
        response = requests.get(url)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).resize((224, 224))
    except (requests.RequestException, IOError):
        raise HTTPException(status_code=400, detail="Unable to download image from the provided URL.")

@app.post("/classify/")
async def classify_image(
        location: Optional[str] = Form(None),
        classifier: Optional[str] = Form("clip"),
        classes: Optional[str] = Form("nudity, wad, offensive, face-attributes, gore"),
        image: UploadFile = File(None)
):
    if not location and not image:
        raise HTTPException(status_code=400, detail="Please provide either 'location' or 'image'.")

    classes = classes.strip(' ').split(', ')
    
    if image:
        pil_image = await get_image_from_upload(image)
    else:
        pil_image = await get_image_from_url(location)

    inputs = processor(text=classes, images=pil_image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    print(outputs.logits_per_image)
    print('/n')
    probs = F.softmax(outputs.logits_per_image, dim=-1).tolist()[0]
    print(F.softmax(outputs.logits_per_image, dim=-1))
    print(probs)
    print('/n')
    label_probs = dict(zip(classes, probs))
    print(zip(classes, probs))
    print(label_probs)
    prediction = max(label_probs, key=label_probs.get)

    result = PredictionResult(
        prediction=prediction,
        label_probabilities=label_probs,
        original_payload={
            "location": location,
            "classifier": "clip"
        }
    )

    return {"result": result}
