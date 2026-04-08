"""
Video file emotion recognition.

Usage:
    python app_video.py <video_path>
    python app_video.py <video_path> -o <output_path>

Examples:
    python app_video.py my_video.mp4
    python app_video.py my_video.mp4 -o result.mp4
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))  # noqa: E402

import cv2  # noqa: E402

from src.config import get_device, CLASS_NAMES, MODEL_PATH  # noqa: E402
from src.preprocessing import detect_faces  # noqa: E402
from src.inference import load_model, classify_faces  # noqa: E402

def process_video(video_path, output_path="output_video.mp4"):
    device = get_device()
    model = load_model(MODEL_PATH, device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, (frame_width, frame_height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (255, 255, 255)
    font_thickness = 2

    print('Analyzing video. Depending on the video length, this might take a while.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces, face_boxes = detect_faces(frame)

        if faces:
            emotions = classify_faces(model, faces, CLASS_NAMES, device)

            for (x, y, w, h), emotion in zip(face_boxes, emotions):
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), font, font_scale, font_color, font_thickness)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Processing complete. Output saved to:", output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Emotion recognition on a video file.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("-o", "--output", default="output_video.mp4", help="Path to the output file (default: output_video.mp4)")
    args = parser.parse_args()
    process_video(args.video, args.output)
