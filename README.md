# ExpressifAI

A webcam demo that recognizes a person's action and facial emotion and generates a short semantic description using the Gemini API. It uses CLIP for zero-shot action recognition and a DenseNet-161 model for emotion classification.

What it does
- Captures frames from your webcam.
- Recognizes actions using CLIP (matching frames against a set of action captions).
- Detects facial emotion with a DenseNet-161 classifier (7 labels).
- Calls the Gemini API to produce a natural-language description; falls back to a simple template if no API key is provided.

Quick setup
1. Install Python 3.10+ and create a virtual environment:

   python -m venv .venv
   .venv\Scripts\Activate.ps1  # PowerShell

2. Install required packages (example):

   pip install torch torchvision opencv-python pillow git+https://github.com/openai/CLIP.git gradio python-dotenv google-generativeai

   Note: If you have a CUDA GPU and want GPU acceleration, install the appropriate `torch`/`torchvision` build from https://pytorch.org/.

3. Add your Gemini API key to a `.env` file in the project root:

   GEMINI_API_KEY=your_api_key_here

Run (recommended)
- The preferred entry point is the Gradio browser UI which streams your webcam and shows predictions:

  python webcam_recognition.py

- `main.py` is a simpler OpenCV-based demo that overlays predictions on a window and can be used as an alternative, but the Gradio UI (`webcam_recognition.py`) is the recommended way to run the project. Press `q` to quit the OpenCV window when using `main.py`.

Notes
- The emotion model weights and CLIP model will be downloaded automatically on first run.
- If `GEMINI_API_KEY` is missing, the app uses a short template description instead of calling the API.

Troubleshooting
- If the webcam does not open, check camera index and ensure no other app is using the camera.
- For best performance, run on a machine with a CUDA-capable GPU and compatible PyTorch.
