import gradio as gr
import torch
import cv2
from PIL import Image
from action_model import load_action_model, recognize_action_single
from emotion_model import load_emotion_model, recognize_emotion_single
from semantic_gen import generate_semantic_description

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
action_model, action_preprocess = load_action_model(device)
emotion_model, emotion_preprocess = load_emotion_model(device)

# Process webcam frames for action recognition and emotion detection
def process_webcam(frame):
    # Gradio provides RGB, but action model expects BGR OpenCV frame
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Get action prediction (pass preprocess)
    action = recognize_action_single(frame_bgr, device, action_model, action_preprocess)
    
    # Convert the frame to PIL image for emotion detection
    pil_image = Image.fromarray(frame)
    
    # Get emotion prediction
    emotion = recognize_emotion_single(pil_image, device, emotion_model, emotion_preprocess)
    
    # Generate semantic description
    output_text = generate_semantic_description(emotion, action)
    
    return frame, f"Action: {action}\nEmotion: {emotion}", output_text

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
            output_box = gr.Textbox(label="Raw Predictions", lines=4)
            
            gr.HTML("<h2 style='font-size: 28px; font-family: Arial, sans-serif; margin-bottom: 10px;'>Semantic Description</h2>")
            terminal_box = gr.Textbox(label="Generated Description", lines=12)
    
    input_img.stream(
        process_webcam,
        inputs=input_img,
        outputs=[webcam_display, output_box, terminal_box],
        stream_every=0.1,
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