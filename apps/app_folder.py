"""
Batch processing: image folder → CSV with emotion scores.

Usage:
    python app_folder.py <folder_path>
    python app_folder.py <folder_path> -o <csv_path>

Examples:
    python app_folder.py ./images/
    python app_folder.py ./images/ -o results.csv
"""

import os
import cv2
import csv

from src.config import get_device, CLASS_NAMES, MODEL_PATH
from src.preprocessing import detect_faces
from src.inference import load_model, classify_with_scores


def main(folder_path, csv_file='test.csv'):
    device = get_device()
    model = load_model(MODEL_PATH, device)

    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    csv_dictionary = []

    for img_name in os.listdir(folder_path):
        if os.path.splitext(img_name)[1].lower() not in valid_extensions:
            continue

        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            print(f"Warning: Could not read {img_path}, skipping.")
            continue

        faces, face_boxes = detect_faces(img)

        if not faces:
            print(f"Warning: No face detected in {img_path}, skipping.")
            continue

        for face in faces:
            scores = classify_with_scores(model, face, CLASS_NAMES, device)
            scores['filepath'] = img_path
            csv_dictionary.append(scores)

    fields = ['filepath'] + CLASS_NAMES

    with open(csv_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_dictionary)

    print(f"File saved as '{csv_file}'.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Batch emotion recognition on a folder of images.")
    parser.add_argument("folder", help="Path to the folder containing images")
    parser.add_argument("-o", "--output", default="test.csv", help="Path to the CSV output file (default: test.csv)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise FileNotFoundError(f'The directory {args.folder} was not found.')

    print('Folder found. Processing...')
    main(args.folder, args.output)
