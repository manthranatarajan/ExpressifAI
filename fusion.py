def generate_output(emotion, action):
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}
    emotion_text = emotion_dict.get(emotion, "Unknown Emotion")
    action_text = ", ".join(action) if action else "Unknown Action"

    return f"Detected Emotion: {emotion_text}, \n\nDetected Actions: {action_text}"
