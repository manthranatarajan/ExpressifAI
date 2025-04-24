import cv2
import torch
from webcam_loader import setup_webcam, get_frame
from action_model import load_action_model, recognize_action_single

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_action_model(device)

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
    last_prediction = "Analyzing..."

    while True:
        frame = get_frame(cap)
        if frame is None:
            break

        frame_count += 1
        if frame_count % predict_every_n_frames == 0:
            last_prediction = recognize_action_single(frame, device, model)

        # Display prediction on video
        cv2.putText(frame, f"Action: {last_prediction}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
