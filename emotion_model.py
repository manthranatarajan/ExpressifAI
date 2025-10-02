import torch
from torchvision import models, transforms
from PIL import Image

EMOTION_LABELS = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

def load_emotion_model(device):
    """Loads the pre-trained DenseNet-161 emotion detection model and moves it to the specified device."""
    model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
    model.classifier = torch.nn.Linear(model.classifier.in_features, 7)  # Adjust for 7 emotions
    model.to(device)
    model.eval()

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return model, preprocess

def recognize_emotion_single(pil_image, device, model, preprocess):
    """Recognizes emotion from a single PIL image."""
    if not isinstance(pil_image, Image.Image):
        raise TypeError(f"Expected a PIL Image, but got {type(pil_image)}")

    tensor_frame = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(tensor_frame)
        predicted_emotion = outputs.argmax(-1).item()
    return EMOTION_LABELS.get(predicted_emotion, "Unknown")
