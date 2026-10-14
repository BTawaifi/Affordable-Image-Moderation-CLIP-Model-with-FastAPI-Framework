# CLIP (Contrastive Language-Image Pretraining) is a model developed by OpenAI that enables joint understanding of images and text. It can process both images and text simultaneously and learn to associate them in a meaningful way. The model has been pretrained on a large corpus of image and text data and can be fine-tuned for various downstream tasks.
# In the case of CLIP, text is used to provide textual descriptions or prompts that help the model understand and reason about the content of images. The model can process natural language queries, instructions, or descriptions and use them to perform tasks such as image classification, object detection, and image generation.
# During training, CLIP learns to align the representations of images and their corresponding textual descriptions in a shared embedding space. This allows the model to compute similarity scores between images and text based on their embeddings. By leveraging this shared embedding space, CLIP can understand the semantic relationships between images and text, enabling tasks like zero-shot image classification, where the model can classify images based on textual descriptions without seeing any labeled image examples during training.
# In summary, CLIP combines the power of visual and textual understanding by learning joint representations for images and text, allowing it to perform a wide range of tasks that involve the understanding of both modalities.

import asyncio
from typing import Dict
import aio_pika
from fastapi import FastAPI, File, Body
from PIL import Image
from pydantic import BaseModel
import io
import json
from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F
import requests

app = FastAPI()

model_name = "openai/clip-vit-base-patch32"
processor = CLIPProcessor.from_pretrained(model_name)
model = CLIPModel.from_pretrained(model_name)
classes = ["nsfw", "sfw"]

class PredictionResult(BaseModel):
    clipPrediction: str
    clipLabelProbabilities: Dict[str, float]

class ChannelManager:
    def __init__(self):
        self.channel = None

    async def get_channel(self):
        if not self.channel:
            connection = await aio_pika.connect_robust("amqp://guest:guest@localhost")
            self.channel = await connection.channel()

            # Declare the classification_result_queue and bind it to the classification_result_exchange
            classification_result_queue = await self.channel.declare_queue("classification_result_queue", durable=True)
            classification_result_exchange = await self.channel.declare_exchange("classification_result_exchange", aio_pika.ExchangeType.DIRECT)
            await classification_result_queue.bind(classification_result_exchange, routing_key="classification_result_queue")

            await self.channel.set_qos(prefetch_count=1)

        return self.channel

channel_manager = ChannelManager()

async def process_message(message: aio_pika.IncomingMessage):
    try:
            channel = await channel_manager.get_channel()
            data = json.loads(message.body.decode())

            # print(data)
            classes = data.get("classes", "nsfw, sfw")
            classes= classes.strip(' ').split(', ')

            print(data["location"])
            if "location" not in data:
                print(f"Error: 'location' key missing in the message: {data}")
                await message.nack(requeue=False)  # Send reject message
                return

            location = data["location"]

            # Download the image from the file server
            response = requests.get('http://localhost:5030'+location)
            response.raise_for_status()
            image_data = response.content

            # Resize the image
            pil_image = Image.open(io.BytesIO(image_data)).resize((224, 224))

            # Perform image classification using CLIP model
            inputs = processor(text=classes, images=pil_image, return_tensors="pt", padding=True)
            # Process the input text and image using the CLIPProcessor
            # - "text=classes" specifies the text input as the provided list of classes
            # - "images=pil_image" specifies the image input as the provided PIL image
            # - "return_tensors='pt'" specifies that the inputs should be returned as PyTorch tensors
            # - "padding=True" pads the text inputs to the maximum length if needed

            # The resulting 'inputs' object contains the processed inputs ready to be passed to the CLIPModel for inference

            outputs = model(**inputs)
            # Pass the processed inputs to the CLIPModel for inference
            # - The double asterisks (**) unpacks the 'inputs' dictionary and passes its contents as keyword arguments
            #   to the model's forward method
            # - The CLIPModel performs forward propagation using the provided inputs and produces the outputs

            # Compute label probabilities
            probs = F.softmax(outputs.logits_per_image, dim=-1).tolist()[0]
            # Apply softmax function to convert logits into probabilities
            # along the last dimension (dim=-1) which represents the classes
            # Convert the resulting tensor to a Python list
            # Take the first (and only) element since we have a single image

            label_probs = dict(zip(classes, probs))
            # Create a dictionary by pairing each class label from the "classes" list
            # with its corresponding probability from the "probs" list

            pred_label = max(label_probs, key=label_probs.get)
            # Find the key (class label) in the "label_probs" dictionary
            # that corresponds to the maximum value (probability)
            # This determines the predicted label with the highest probability

            # Create the result payload
            result = { 
                "pattern": "classification_result_queue",
                "data": {
                "clipPrediction": pred_label,
                "clipLabelProbabilities": label_probs,
                "originalPayload": data,
            }}

            result_json = json.dumps(result)

            try:
                # Publish the classification result to the exchange
                exchange = await channel.get_exchange('classification_result_exchange')
                await exchange.publish(
                    aio_pika.Message(body=result_json.encode()),
                    routing_key="classification_result_queue",
                )
                await message.ack()  # Acknowledge successful processing of the message
            except Exception as e:
                print(f"Error occurred while publishing message: {e}")
                await message.nack(requeue=False)  # Send reject message
    except Exception as e:
        print(f"Error occurred while processing message: {e}")
        await message.nack(requeue=False)  # Send reject message
  

def create_application():
    async def main():
        channel = await channel_manager.get_channel()
        queue = await channel.declare_queue("classification", durable=True)
        await queue.consume(process_message)

    asyncio.get_event_loop().create_task(main())

create_application()

#uvicorn clip_classifier_consumer:app --host 0.0.0.0 --port 6005 --workers 1 --reload