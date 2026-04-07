# CNN for Real-Time Facial Emotion Recognition

CNN-based emotion recognition using **ResNet-50**, trained on the **RAF-DB** dataset. The project supports three application modes: live webcam, video files, and batch processing of image folders.

---

## Setup

### Prerequisites

- Python 3.10+
- Webcam (for `app_webcam.py`)
- GPU optional, auto-detected (CUDA / Apple MPS)

### Installation

```bash
git clone https://github.com/Benni-Carpenter/live-emotion-recognition.git
cd live-emotion-recognition

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Pretrained Model

The pretrained model (`models/trained_model_rafdb.pth`, ~90 MB) is included directly in this repository. No separate download is needed — it will be available after cloning.

> **Note:** Due to the large file size, cloning may take a moment longer than usual.

If you want to retrain the model yourself, obtain the RAF-DB dataset and place the zip file at the project root as `RafDB-Dataset.zip`. The training script will automatically extract it on first run. The expected structure after extraction is:

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

Then run:

```bash
python -m src.training
```

## Usage

All apps are run from the project root:

### Live Webcam

```bash
python -m apps.app_webcam              # Default: classify every 2 seconds
python -m apps.app_webcam -i 5         # Classify every 5 seconds
```

Press **Q** to quit.

### Video File

```bash
python -m apps.app_video my_video.mp4
python -m apps.app_video my_video.mp4 -o output.mp4
```

### Image Folder → CSV

```bash
python -m apps.app_folder ./images/
python -m apps.app_folder ./images/ -o results.csv
```

Produces a CSV file with softmax scores per emotion for each detected face.

## Recognized Emotions

| Class      | Label |
|------------|-------|
| Anger      | 0     |
| Disgust    | 1     |
| Fear       | 2     |
| Happiness  | 3     |
| Sadness    | 4     |
| Surprise   | 5     |

## Architecture

The model is based on **ResNet-50** (without pretrained ImageNet weights), with the final fully-connected layer replaced to output 6 classes. Training uses the Adam optimizer, CrossEntropyLoss, and a Cosine Annealing LR scheduler. Face detection is handled by **dlib** (HOG-based).

## Customizing Training

The main hyperparameters can be adjusted in `src/training.py` when calling `run_training()`:

```python
run_training(device, dataset, batch_size=32, learning_rate=0.001, num_epochs=20)
```

Data augmentations (flip, crop, ColorJitter) are defined in `src/dataset.py`.

## Dependencies

See `requirements.txt`. Key packages:

- `torch` / `torchvision`
- `opencv-python`
- `dlib`
- `Pillow`
- `matplotlib`
- `numpy`
