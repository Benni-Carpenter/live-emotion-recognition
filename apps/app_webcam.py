"""
Live webcam emotion recognition.

Usage:
    python app_webcam.py
    python app_webcam.py -i 3

Options:
    -i, --interval  Seconds between classifications (default: 2)
"""

import cv2
import time

from src.config import get_device, CLASS_NAMES, MODEL_PATH
from src.preprocessing import detect_faces
from src.inference import load_model, classify_faces


def main(interval=2):
    device = get_device()
    model = load_model(MODEL_PATH, device)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return

    start_time = time.time()
    n_seconds = interval

    cached_results = []

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    font_thickness = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Can't receive frame")
            break

        current_time = time.time()
        if current_time - start_time >= n_seconds:
            start_time = current_time

            faces, face_boxes = detect_faces(frame)
            if faces:
                emotions = classify_faces(model, faces, CLASS_NAMES, device)
                cached_results = list(zip(face_boxes, emotions))
            else:
                cached_results = []

        for (x, y, w, h), emotion in cached_results:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), font, font_scale, font_color, font_thickness)

        cv2.imshow('Emotion Classifier', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live webcam emotion recognition.")
    parser.add_argument("-i", "--interval", type=float, default=2,
                        help="Seconds between classifications (default: 2)")
    args = parser.parse_args()
    main(args.interval)
