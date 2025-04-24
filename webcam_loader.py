import cv2

def setup_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam successfully opened.")
    return cap

def get_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        return None
    return frame
