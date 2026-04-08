# CNN for Real-Time Facial Emotion Recognition

CNN-based emotion recognition using **ResNet-50**, trained on the **RAF-DB** dataset. Detects faces in real time and classifies them into one of six emotions.

Three application modes are supported:

| Mode | Script |
|------|--------|
| Live webcam | `apps/app_webcam.py` |
| Video file | `apps/app_video.py` |
| Image folder → CSV | `apps/app_folder.py` |

---

## Project Structure

```
cv-fer-project/
├── apps/
│   ├── app_webcam.py       # Live webcam inference
│   ├── app_video.py        # Video file inference
│   └── app_folder.py       # Batch folder inference → CSV
├── models/
│   └── trained_model_rafdb.pth
├── src/
│   ├── config.py           # Paths, hyperparameters, device selection
│   ├── dataset.py          # RAF-DB dataset loader with augmentation
│   ├── inference.py        # Model loading and face classification
│   ├── model.py            # ResNet-50 architecture definition
│   ├── preprocessing.py    # Face detection (dlib HOG)
│   └── training.py         # Training loop
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- Webcam (for `app_webcam.py`)
- GPU optional — CUDA and Apple MPS are auto-detected

### Installation

```bash
git clone https://github.com/Benni-Carpenter/live-emotion-recognition.git
cd live-emotion-recognition

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Pretrained Model

The pretrained model (`models/trained_model_rafdb.pth`, ~90 MB) is included in the repository. No separate download is needed after cloning.

> **Note:** Due to the large file size, cloning may take a moment longer than usual.

---

## Usage

All apps can be run from any directory — paths are resolved automatically.

### Live Webcam

```bash
python apps/app_webcam.py              # Classify every 2 seconds (default)
python apps/app_webcam.py -i 1         # Classify every 1 second
```

Press **Q** to quit.

### Video File

```bash
python apps/app_video.py my_video.mp4
python apps/app_video.py my_video.mp4 -o output.mp4
```

### Image Folder → CSV

```bash
python apps/app_folder.py ./images/
python apps/app_folder.py ./images/ -o results.csv
```

Produces a CSV with softmax confidence scores per emotion for each detected face.

---

## Recognized Emotions

`anger` · `disgust` · `fear` · `happiness` · `sadness` · `surprise`

---

## Architecture

- **Backbone:** ResNet-50 (trained from scratch, no ImageNet weights)
- **Output:** Final FC layer replaced to output 6 classes
- **Optimizer:** Adam with Cosine Annealing LR scheduler
- **Loss:** CrossEntropyLoss
- **Face detection:** dlib HOG-based detector

---

## Retraining

To retrain the model, obtain the RAF-DB dataset and place the zip at the project root as `RAF-DB.zip`. The training script extracts it automatically on first run.

Expected structure after extraction:

```
RafDB-Dataset/
├── train/
│   ├── anger/
│   ├── disgust/
│   ├── fear/
│   ├── happiness/
│   ├── sadness/
│   └── surprise/
└── validation/
    └── (same subfolders)
```

Adjust hyperparameters in `src/training.py`:

```python
run_training(device, dataset, batch_size=32, learning_rate=0.001, num_epochs=20)
```

Data augmentations (random flip, crop, ColorJitter) are defined in `src/dataset.py`.

Then run:

```bash
python -m src.training
```

---

## Dependencies

See `requirements.txt`. Key packages:

- `torch` / `torchvision`
- `opencv-python`
- `dlib`
- `Pillow`
- `numpy`
- `matplotlib`
