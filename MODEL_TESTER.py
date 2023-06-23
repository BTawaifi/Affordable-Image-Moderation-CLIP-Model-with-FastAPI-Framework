# Import required libraries
import os
import psutil
import time
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from transformers import CLIPProcessor, CLIPModel
import torch
import torch.nn.functional as F
from PIL import Image
import shutil

# Measure the initial memory usage
start_memory = psutil.virtual_memory().used

# Check for CUDA (GPU); if unavailable, use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Indicate which device is being used
if torch.cuda.is_available():
    print("Using GPU for Processing")
else:
    print("Using CPU for Processing")

# Initialize the name of the pretrained model
model_name = "openai/clip-vit-base-patch32"
# Load the model's processor
processor = CLIPProcessor.from_pretrained(model_name)
# Load the model and move it to the selected device
clip_model = CLIPModel.from_pretrained(model_name).to(device)

def move_incorrect_images(incorrect_images, target_directory):
    # Check if the directory exists; if not, create it
    if not os.path.exists(target_directory):
        os.makedirs(target_directory)
    # Move each incorrectly classified image to the target directory
    for image in incorrect_images:
        shutil.move(image, target_directory)

def evaluate_model(image_directory, test_label, label_list):
    # Initialize arrays for predictions and incorrectly classified images
    predictions = []
    total_time = 0
    incorrect_images = []

    # Retrieve all images from the image directory
    test_images = [
        os.path.join(image_directory, img)
        for img in os.listdir(image_directory)
        if img.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Set the labels for all test images to be the test_label
    test_labels = [test_label for _ in range(len(test_images))]

    for index, image in enumerate(test_images, start=1):
        # Open and resize the image, then process it with the model's processor
        pil_image = Image.open(image).resize((224, 224))
        inputs = processor(
            text=label_list, images=pil_image, return_tensors="pt", padding=True
        )

        # Move inputs to the selected device
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        # Track the start time
        start_time = time.time()

        # Generate model outputs for the inputs
        outputs = clip_model(**inputs)
        # Convert outputs to probabilities
        probs = F.softmax(outputs.logits_per_image, dim=-1).tolist()[0]
        # Create a dictionary of labels and their corresponding probabilities
        label_probs = dict(zip(label_list, probs))

        # Track the end time
        end_time = time.time()
        # Calculate total processing time
        total_time += end_time - start_time

        # Determine the prediction with the highest probability
        prediction = max(label_probs, key=label_probs.get)
        # Add the prediction to the predictions array
        predictions.append(prediction)

        # Display the image being processed
        print(f"Processing image {index} of {len(test_images)}: {image}")

        # Check if the prediction matches the test label; if not, add to incorrect_images
        if prediction != test_label:
            incorrect_images.append(image)

    # Calculate evaluation metrics
    accuracy = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average="weighted")
    recall = recall_score(test_labels, predictions, average="weighted")
    f1 = f1_score(test_labels, predictions, average="weighted")
    avg_time_per_image = total_time / len(test_images)
    end_memory = psutil.virtual_memory().used
    memory_used = end_memory - start_memory

    # Display the evaluation metrics and memory usage
    print(f"Model accuracy: {accuracy * 100:.2f}%")
    print(f"Model precision: {precision * 100:.2f}%")
    print(f"Model recall: {recall * 100:.2f}%")
    print(f"Model F1-score: {f1 * 100:.2f}%")
    print(f"Average time per image: {avg_time_per_image:.5f} seconds")
    print(f"Memory used: {memory_used / (1024**2):.2f} MB")

    # Display the confusion matrix
    cm = confusion_matrix(test_labels, predictions, labels=label_list)
    print("Confusion matrix:")
    print(cm)

    # Display the incorrectly classified images
    if not incorrect_images: 
        print("No incorrect images found.") 
    else: print("Incorrectly classified images:") 
        print(incorrect_images)

    # Move incorrectly classified images to a separate directory
    target_directory = image_directory + "/Incorrect"
    if incorrect_images:
        move_incorrect_images(incorrect_images, target_directory)

# Set the image directory, test label, and list of possible labels
image_directory = "./nsfw"
test_label = "nsfw"
label_list = ["nsfw", "sfw"]

# Evaluate the model
evaluate_model(image_directory, test_label, label_list)