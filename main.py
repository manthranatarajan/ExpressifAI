import cv2
import torch
from webcam_loader import setup_webcam, get_frame
from emotion_model import detect_emotion
from action_model import load_action_model, recognize_action
from fusion import generate_output
from semantic_gen import generate_semantic_description
import textwrap
from action_model import load_action_model, recognize_action_single


def draw_text_wrapped(img, text, start_pos, font, font_scale, color, thickness, line_spacing=10, max_width=50):
    x, y = start_pos
    wrapped_text = textwrap.wrap(text, width=max_width)

    for i, line in enumerate(wrapped_text):
        y_line = y + i * (int(font_scale * 30) + line_spacing)  # 30 is approx height multiplier
        cv2.putText(img, line, (x, y_line), font, font_scale, color, thickness, cv2.LINE_AA)


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
        cv2.namedWindow('Webcam Feed', cv2.WINDOW_NORMAL) #make it resizable
        _, _, win_w, win_h = cv2.getWindowImageRect("Webcam Feed") #get current window size
        resized_frame = cv2.resize(frame, (win_w, win_h)) #resize frame to match window
        # cv2.putText(resized_frame, output_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # draw on resized_frame instead of original frame 
        # cv2.putText(resized_frame, output_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        draw_text_wrapped(
            resized_frame,
            output_text,
            start_pos=(20, 50),
            font=cv2.FONT_HERSHEY_SIMPLEX,
            font_scale=0.6,
            color=(255, 255, 255),
            thickness=1,
            line_spacing=5,
            max_width=50 
        )

        #display resized frame
        cv2.imshow("Webcam Feed", resized_frame)        
        
        # Display prediction on video
        #cv2.putText(frame, f"Action: {last_prediction}", (10, 50),
        #            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        #cv2.imshow('Webcam Feed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
