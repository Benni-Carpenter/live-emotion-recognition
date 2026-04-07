"""
Preprocessing: face detection (using dlib) and Gabor filter.
"""

import cv2
import dlib
import numpy as np

from src.config import GABOR_KSIZE, GABOR_SIGMA, GABOR_THETA, GABOR_LAMBDA, GABOR_GAMMA, GABOR_PSI


_face_detector = dlib.get_frontal_face_detector()


def detect_faces(img):
    """Detects faces in a BGR image. Returns (faces, boxes)."""
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = _face_detector(img_grayscale)
    faces_preprocessed = []
    face_boxes = []

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)

        if w > 0 and h > 0:
            face_only = img[y:y+h, x:x+w]
            faces_preprocessed.append(face_only)
            face_boxes.append((x, y, w, h))

    return faces_preprocessed, face_boxes


def apply_gabor_filter(image, ksize=GABOR_KSIZE, sigma=GABOR_SIGMA,
                       theta=GABOR_THETA, lambd=GABOR_LAMBDA,
                       gamma=GABOR_GAMMA, psi=GABOR_PSI):
    """Applies a Gabor filter and normalizes the result."""
    gabor_kernel = cv2.getGaborKernel(
        ksize=(ksize, ksize), sigma=sigma, theta=theta,
        lambd=lambd, gamma=gamma, psi=psi, ktype=cv2.CV_32F
    )
    filtered_image = cv2.filter2D(image.astype(np.float32), -1, gabor_kernel)
    filtered_image_normalized = cv2.normalize(
        filtered_image, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    return filtered_image_normalized
