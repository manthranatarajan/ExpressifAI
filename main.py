import cv2
import torch
from PIL import Image
from webcam_loader import setup_webcam, get_frame
from action_model import load_action_model, recognize_action_single
from emotion_model import load_emotion_model, recognize_emotion_single
from semantic_gen import generate_semantic_description

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    action_model = load_action_model(device)
    action_model, action_preprocess = load_action_model(device)
    emotion_model, emotion_preprocess = load_emotion_model(device)

    # Setup webcam
    cap = setup_webcam()
    if not cap:
        return

    # Create resizable window
    cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Webcam Feed', 1200, 600)

    # Initialize frame counter and prediction buffer
    frame_count = 0
    predict_every_n_frames = 15  # Predict every half second (assuming ~30 FPS)
    last_action_prediction = "Analyzing..."
    last_emotion_prediction = "Detecting..."
    last_semantic_output = "Generating description..."

    while True:
        frame = get_frame(cap)
        if frame is None:
            break

        frame_count += 1
        if frame_count % predict_every_n_frames == 0:
            # Convert BGR frame to RGB and then to a PIL Image for the models
            pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            last_action_prediction = recognize_action_single(frame, device, action_model, action_preprocess)
            last_emotion_prediction = recognize_emotion_single(pil_image, device, emotion_model, emotion_preprocess)
            last_semantic_output = generate_semantic_description(last_emotion_prediction, last_action_prediction)

        # Display predictions on video
        cv2.putText(frame, f"Action: {last_action_prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Emotion: {last_emotion_prediction}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        y0 = 100
        for i, line in enumerate(last_semantic_output.split("\n")):
            y = y0 + i * 25
            cv2.putText(frame, line, (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)

        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
