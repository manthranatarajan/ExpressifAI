import cv2
import torch
from action_model import load_action_model

def test_webcam_and_model():
    # Test webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    print("Webcam successfully opened.")
    cap.release()

    # Test action recognition model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = load_action_model(device)
        print("Action recognition model loaded successfully.")
    except Exception as e:
        print(f"Error loading action recognition model: {e}")

# Run the test
test_webcam_and_model()
