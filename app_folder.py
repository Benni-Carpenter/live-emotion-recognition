"""
Batch-Verarbeitung: Ordner mit Bildern → CSV mit Emotionsscores.

Aufruf:
    python app_folder.py <ordner_pfad>
    python app_folder.py <ordner_pfad> -o <csv_pfad>

Beispiele:
    python app_folder.py ./bilder/
    python app_folder.py ./bilder/ -o ergebnis.csv
"""

import os
import cv2
import csv

from config import get_device, CLASS_NAMES, MODEL_PATH
from preprocessing import detect_faces
from inference import load_model, classify_with_scores


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
    parser = argparse.ArgumentParser(description="Batch-Emotionserkennung auf einem Ordner mit Bildern.")
    parser.add_argument("folder", help="Pfad zum Ordner mit Bildern")
    parser.add_argument("-o", "--output", default="test.csv", help="Pfad zur CSV-Ausgabedatei (default: test.csv)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        raise FileNotFoundError(f'The directory {args.folder} was not found.')

    print('Folder found. Processing...')
    main(args.folder, args.output)
