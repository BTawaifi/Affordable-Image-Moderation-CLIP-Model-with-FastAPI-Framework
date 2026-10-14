import os
from typing import Dict
from fastapi import FastAPI, File, Body
from PIL import Image
from pydantic import BaseModel
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
from sqlalchemy import Column, Integer, String, create_engine, ForeignKey, Float, Boolean, DateTime
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from nudenet import NudeClassifier
from torchvision import transforms, models
import numpy as np
import io
import torch
from urllib.parse import urlparse
import requests
import uuid

app = FastAPI()

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
clip_model = CLIPModel.from_pretrained(model_name)
nude_classifier = NudeClassifier()
resnet_model = models.resnet50(pretrained=True)
resnet_model.eval()

# Transformation for ResNet
transform = transforms.Compose([            
    transforms.Resize(256),                    
    transforms.CenterCrop(224),                
    transforms.ToTensor(),                     
    transforms.Normalize(                      
    mean=[0.485, 0.456, 0.406],                
    std=[0.229, 0.224, 0.225]                  
)])

# SQLAchemy setup
Base = declarative_base()

class ClassificationResult(Base):
    __tablename__ = "classification_result"
    id = Column(Integer, primary_key=True)
    classifier = Column(String)
    location = Column(String)
    prediction = Column(String)
    moderated = Column(Boolean, default=False)
    createdAt = Column(DateTime)
    modifiedAt = Column(DateTime)
    labelProbability = relationship("LabelProbability", back_populates="classificationResult")

class LabelProbability(Base):
    __tablename__ = "label_probability"

    id = Column(Integer, primary_key=True)
    classificationResultId = Column(Integer, ForeignKey('classification_result.id'))
    classificationResult = relationship("ClassificationResult", back_populates="labelProbability")
    label = Column(String)
    probability = Column(Float)

engine = create_engine("mysql+pymysql://username:***REDACTED***@localhost/dbname")
Session = sessionmaker(bind=engine)

class PredictionResult(BaseModel):
    prediction: str
    labelProbabilities: Dict[str, float]
    originalPayload: dict

@app.post("/classify_image/")
async def classify_image(location: str, classifier: str = Body(default="clip"), classes: str = Body(default="nudity, wad, offensive, face-attributes, gore")):
    classes = classes.strip(' ').split(', ')

    if urlparse(location).scheme in ['http', 'https']:
        response = requests.get(location)
        response.raise_for_status()
        image_data = response.content
        pil_image = Image.open(io.BytesIO(image_data)).resize((224, 224))
    else:
        pil_image = Image.open(location).resize((224, 224))

    if classifier.lower() == "nudenet":
        opencv_image = np.array(pil_image)
        opencv_image = opencv_image[:, :, ::-1].copy()
        result = nude_classifier.classify(opencv_image)
        prediction = max(result[0], key=result[0].get)
        label_probs = result[0]
    elif classifier.lower() == "resnet":
        image_tensor = transform(pil_image).unsqueeze(0)
        output = resnet_model(image_tensor)
        probs = F.softmax(output, dim=-1).detach().numpy()[0]
        label_probs = dict(zip(classes, probs))
        prediction = max(label_probs, key=label_probs.get)
    else:
        inputs = processor(text=classes, images=pil_image, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        probs = F.softmax(outputs.logits_per_image, dim=-1).tolist()[0]
        label_probs = dict(zip(classes, probs))
        prediction = max(label_probs, key=label_probs.get)

    result = PredictionResult(
        prediction = prediction,
        labelProbabilities = label_probs,
        originalPayload = {
            "location": location,
            "classifier": classifier
        }
    )

    # Compress and save image to a local file
    compressed_image_filename = f"{uuid.uuid4().hex}.jpeg"
    compressed_image_path = os.path.join("/uploads", compressed_image_filename)
    pil_image.save(compressed_image_path, "JPEG", quality=90)

    session = Session()
    moderation_result = ClassificationResult(
        classifier = result.originalPayload["classifier"],
        location = compressed_image_path,  # Here we use the path of the compressed image
        prediction = result.prediction,
        moderated = False
    )
    session.add(moderation_result)
    session.commit()

    label_probabilities = [
        LabelProbability(classificationResultId = moderation_result.id, label = label, probability = prob) 
        for label, prob in result.labelProbabilities.items()
    ]
    session.bulk_save_objects(label_probabilities)
    session.commit()

    return {"message": "Classification result saved successfully!"}


#uvicorn clip_rest_full:app --host 0.0.0.0 --port 6002 --workers 1 --reload 