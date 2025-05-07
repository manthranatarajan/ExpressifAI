import gradio as gr
import torch
import cv2
from PIL import Image
from action_model import load_action_model, recognize_action_single
from emotion_model import detect_emotion
from fusion import generate_output

# Emotion label mapping
EMOTION_LABELS = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load action model
action_model, preprocess = load_action_model(device)

# Process webcam frames for action recognition and emotion detection
def process_webcam(frame):
    # Convert frame from RGB to BGR
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Get action using recognize_action_single and wrap in a list
    action = recognize_action_single(frame_bgr, device, (action_model, preprocess))
    actions = [action]  # generate_output expects a list
    
    # Convert the frame to PIL image for emotion detection
    pil_image = Image.fromarray(frame)
    
    # Detect emotion from the frame
    emotion_idx = detect_emotion(pil_image)
    
    # Generate output based on emotion and action
    output_text = generate_output(emotion_idx, actions)     # Pass index, not label
    
    return frame, output_text, output_text  # Return frame for webcam display, and outputs

# Create the Gradio interface
with gr.Blocks() as demo:
    with gr.Row():
        # Left side with webcam input
        with gr.Column(elem_id="left-panel", min_width=400):
            gr.HTML("<h2 style='font-size: 28px; font-family: Arial, sans-serif; margin-bottom: 10px;'>Webcam</h2>")
            input_img = gr.Image(
                sources=["webcam"],
                streaming=True,
                label="Input",
                type="numpy",
                height=600
            )
            webcam_display = gr.Image(label="Webcam Output", type="numpy", height=600)
        
        # Right side with Output section
        with gr.Column(scale=1, elem_id="right-panel", min_width=200):
            gr.HTML("<h2 style='font-size: 28px; font-family: Arial, sans-serif; margin-bottom: 10px;'>Output</h2>")
            output_box = gr.Textbox(label="Output", lines=4)
            
            gr.HTML("<h2 style='font-size: 28px; font-family: Arial, sans-serif; margin-bottom: 10px;'>Terminal</h2>")
            terminal_box = gr.Textbox(label="Terminal Output", lines=12)
    
    input_img.stream(
        process_webcam,
        inputs=input_img,
        outputs=[webcam_display, output_box, terminal_box],
        time_limit=15,
        stream_every=0.1,
        concurrency_limit=30
    )

# CSS settings
demo.css = """
#left-panel {
    background-color: #FFC3C3;
    padding: 20px;
    border-radius: 12px;
}

#right-panel {
    background-color: #3395F1;
    padding: 20px;
    border-radius: 12px;
    height: 700px; /* Set equal height to match left panel */
}

#right-panel .gradio-textbox {
    flex-grow: 1;
    height: 50%;
}
"""

if __name__ == "__main__":
    demo.launch()