import os
from typing import Dict, List, Union
from pydantic import BaseModel, HttpUrl, ValidationError, AnyUrl
from fastapi import FastAPI, File, Body, HTTPException
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sqlalchemy import Column, Integer, String, create_engine, ForeignKey, Float, Boolean, DateTime
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import numpy as np
import io
from urllib.parse import urlparse
import requests
import uuid
import torch.nn.functional as F

app = FastAPI()

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)

# SQLAchemy setup
Base = declarative_base()


class ClassificationResult(Base):
    __tablename__ = "classification_result"
    id = Column(Integer, primary_key=True)
    classifier = Column(String)
    location = Column(String)
    prediction = Column(String)
    moderated = Column(Boolean, default=False)
    labelProbability = relationship(
        "LabelProbability", back_populates="classificationResult")


class LabelProbability(Base):
    __tablename__ = "label_probability"

    id = Column(Integer, primary_key=True)
    classificationResultId = Column(
        Integer, ForeignKey('classification_result.id'))
    classificationResult = relationship(
        "ClassificationResult", back_populates="labelProbability")
    label = Column(String)
    probability = Column(Float)


engine = create_engine(
    "mysql+pymysql://root:***REDACTED***@localhost/image_mod_db")
Session = sessionmaker(bind=engine)


class PredictionResult(BaseModel):
    prediction: str
    labelProbabilities: Dict[str, float]
    originalPayload: dict


class ImageClassificationRequest(BaseModel):
    location: Union[HttpUrl, str]
    classifier: str = "clip"
    classes: str = "nudity, wad, offensive, face-attributes, gore"


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {"detail": exc.detail}, exc.status_code


@app.exception_handler(ValidationError)
async def validation_exception_handler(request, exc):
    return {"detail": exc.errors()}, 422


def validate_image_location(location: str) -> Image.Image:
    try:
        parsed_url = urlparse(location)
        if parsed_url.scheme in ['http', 'https']:
            response = requests.get(location)
            response.raise_for_status()
            image_data = response.content
            pil_image = Image.open(io.BytesIO(image_data)).resize((224, 224))
        else:
            local_path = os.path.abspath(location)
            if not os.path.exists(local_path) or not os.path.isfile(local_path):
                raise FileNotFoundError
            pil_image = Image.open(local_path).resize((224, 224))
        return pil_image
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Image file not found")
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="Failed to access the remote image")
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to load the image")


@app.post("/classify_image/")
async def classify_image(request: ImageClassificationRequest):
    location = request.location
    classifier = request.classifier
    classes = request.classes

    try:
        classes = classes.strip(' ').split(', ')
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid classes")

    try:
        pil_image = validate_image_location(location)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to load the image")

    try:
        inputs = processor(text=classes, images=pil_image,
                           return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        probs = F.softmax(outputs.logits_per_image, dim=-1).tolist()[0]
        label_probs = dict(zip(classes, probs))
        prediction = max(label_probs, key=label_probs.get)
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during classification")

    result = PredictionResult(
        prediction=prediction,
        labelProbabilities=label_probs,
        originalPayload={
            "location": location,
            "classifier": classifier
        }
    )

    session = Session()
    moderation_result = ClassificationResult(
        classifier=result.originalPayload["classifier"],
        location=result.originalPayload["location"],
        prediction=result.prediction,
        moderated=False
    )
    session.add(moderation_result)
    session.commit()

    label_probabilities = [
        LabelProbability(
            classificationResultId=moderation_result.id, label=label, probability=prob)
        for label, prob in result.labelProbabilities.items()
    ]
    session.bulk_save_objects(label_probabilities)
    session.commit()

    return {"result": result.dict()}
