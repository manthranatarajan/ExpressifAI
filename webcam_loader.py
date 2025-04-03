import cv2

def setup_webcam():
    cap = cv2.VideoCapture(0)  # Open the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    print("Webcam successfully opened.")
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None
    return frame
