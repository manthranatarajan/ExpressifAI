import torch
from torchvision import models, transforms

# Load pre-trained DenseNet-161 emotion detection model
emotion_model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
emotion_model.classifier = torch.nn.Linear(emotion_model.classifier.in_features, 7)  # Adjust for 7 emotions
emotion_model.eval()

# Transformation for input frames
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def detect_emotion(frame):
    tensor_frame = transform(frame).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = emotion_model(tensor_frame)
        predicted_emotion = outputs.argmax(-1).item()
    return predicted_emotion
