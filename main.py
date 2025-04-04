import cv2
import torch
from PIL import Image
from webcam_loader import setup_webcam, get_frame
from emotion_model import detect_emotion
from action_model import load_action_model, recognize_action
from fusion import generate_output
from semantic_gen import generate_semantic_description

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    cap = setup_webcam()
    if not cap:
        return

    # Load models
    emotion_action_model = load_action_model(device)

    while True:
        frame = get_frame(cap)
        if frame is None:
            break

        # Convert frame to PIL Image for processing
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # Detect emotion and action
        emotion = detect_emotion(pil_frame)
        action_predictions = recognize_action([frame], device, emotion_action_model)

        # Check if predictions were generated successfully
        if not action_predictions:
            print("No action predictions were generated.")
            output_text = "No actions detected."
        else:
            # generate semantic description
            output_text = generate_semantic_description(action_predictions[0], emotion)
        
        # Generate output text (combine top predicted actions with detected emotion)
        # output_text = generate_output(emotion, action_predictions[0]) 

        # Display the output on the video
        cv2.putText(frame, output_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
