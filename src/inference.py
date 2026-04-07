"""
Inference: load model and classify faces.
"""

import torch
from PIL import Image
import torchvision.transforms as transforms

from src.config import IMAGE_SIZE, CLASS_NAMES, MODEL_PATH, get_device


_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


def load_model(model_path=MODEL_PATH, device=None):
    """Loads the trained model."""
    if device is None:
        device = get_device()
    model = torch.load(model_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def classify_faces(model, faces, class_names=CLASS_NAMES, device=None):
    """Classifies a list of face images (BGR numpy arrays)."""
    if device is None:
        device = get_device()

    predictions = []
    for face in faces:
        face = Image.fromarray(face)
        face = _transform(face).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(face)
        _, predicted = torch.max(output, 1)
        predictions.append(class_names[predicted.item()])

    return predictions


def classify_with_scores(model, img, class_names=CLASS_NAMES, device=None):
    """Classifies a single image and returns per-class scores."""
    if device is None:
        device = get_device()

    img = Image.fromarray(img)
    img = _transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    scores = torch.nn.functional.softmax(output, dim=1)
    class_scores = {class_names[i]: round(score.item(), 2) for i, score in enumerate(scores.squeeze())}
    return class_scores
