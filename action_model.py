import torch
import clip
from PIL import Image
import cv2

STATIONARY_ACTIONS = {
    0: "A person nodding their head in agreement",
    1: "One person is using both of their hands opened up to clap",
    2: "A person laughing out loud",
    3: "A person smiling happily",
    4: "A person sitting still and not moving",
    5: "A person waving by moving the palm of one of their hands with fingers spread out rapidly to say hello",
    6: "A person pointing at something",
    7: "A person looking around curiously",
    8: "A person typing on a keyboard",
    9: "A person reading a book or document",
    10: "A person using a smartphone or tablet",
    11: "A person is praying with their hands joint together",
    12: "A person with glasses on their face is just sitting still",
    13: "Two people are sitting together and talking by opening and closing their mouths",
    14: "Two people are facing each other",
    15: "A person sitting and fixing their hair by runnig their fingers through it",
    16: "A person sitting and scratching their head",
    17: "A person sitting and yawning",
    18: "A person is hiding their face with one hand",
    19: "A person is hiding their face with both hands",
    20: "Their is no face detected in the frame",
    21: "Two people are kissing each other on the lips",
    22: "A person is punching their fist into the palm of their other hand",
    23: "A person is punching the screen with one hand made into a fist",
    24: "A person is punching another person with one hand made into a fist",
    26: "A person giving another person a fist bump with both of them using one hand made into a fist and making contact with each other",
    25: "A person giving another person a hug with both arms around each other",

}


def load_action_model(device):
    """Loads the CLIP model and preprocessing pipeline."""
    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
    return clip_model, preprocess


def recognize_action_single(frame, device, action_model):

    
    try:
        clip_model, preprocess = action_model

        # Convert BGR OpenCV frame to RGB PIL image
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        clip_input = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = clip_model.encode_image(clip_input)
            text_inputs = clip.tokenize(list(STATIONARY_ACTIONS.values())).to(device)
            text_features = clip_model.encode_text(text_inputs)

        # Normalize features
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = image_features @ text_features.T
        pred_index = similarity.argmax(dim=-1).item()

        return STATIONARY_ACTIONS[pred_index]

    except Exception as e:
        print(f"Error in lightweight action recognition: {e}")
        return "Error detecting action"
