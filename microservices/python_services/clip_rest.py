from fastapi import FastAPI, Body, File, Request, UploadFile
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
    labelProbabilities: dict
    originalPayload: dict


@app.post("/classify_image/")
async def classify_image(request: Request, image: UploadFile = File(None)):
    body = await request.json()
    location = body.get("location")
    classifier = body.get("classifier", "clip")
    classes = body.get("classes", "nudity, wad, offensive, face-attributes, gore")

    if not location and not image:
        return {"error": "Please provide either 'location' or 'image'."}

    classes = classes.strip(' ').split(', ')

    if image:
        pil_image = Image.open(io.BytesIO(await image.read())).resize((224, 224))
    elif urlparse(location).scheme in ['http', 'https']:
        response = requests.get(location)
        response.raise_for_status()
        image_data = response.content
        pil_image = Image.open(io.BytesIO(image_data)).resize((224, 224))
    else:
        return {"error": "Invalid image"}

    inputs = processor(text=classes, images=pil_image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)
    probs = F.softmax(outputs.logits_per_image, dim=-1).tolist()[0]
    label_probs = dict(zip(classes, probs))
    prediction = max(label_probs, key=label_probs.get)

    result = PredictionResult(
        prediction=prediction,
        labelProbabilities=label_probs,
        originalPayload={
            "location": location,
            "classifier": classifier
        }
    )

    return {"result": result}
