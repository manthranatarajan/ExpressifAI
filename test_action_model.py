import numpy as np
import torch
from action_model import load_action_model, recognize_action_single

def test_recognize_action():
    # Simulate input frames 
    num_frames = 10
    height, width = 256, 256
    frames = [np.random.randint(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(num_frames)]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the action recognition model
    action_model = load_action_model(device)

    # Perform action recognition
    predictions = recognize_action_single(frames, device, action_model)

    # Print the predictions
    print("Predicted Actions:", predictions)

if __name__ == "__main__":
    test_recognize_action()