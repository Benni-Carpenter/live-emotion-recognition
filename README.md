# cv-fer-project – Facial Emotion Recognition

CNN-basierte Emotionserkennung auf Basis von **ResNet-50**, trainiert auf dem **RAF-DB** Datensatz. Das Projekt unterstützt drei Anwendungsmodi: Live-Webcam, Video-Dateien und Batch-Verarbeitung ganzer Bildordner.

---

## Setup

### Voraussetzungen

- Python 3.10+
- Webcam (für `app_webcam.py`)
- GPU optional, wird automatisch erkannt (CUDA / Apple MPS)

### Installation

```bash
git clone https://github.com/<user>/cv-fer-project.git
cd cv-fer-project

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Trainiertes Modell

Das vortrainierte Modell (`trained_model_rafdb.pth`) ist zu groß für Git. Lade es separat herunter und lege es in `models/` ab – oder trainiere selbst:

```bash
python -m src.training
```

Der Datensatz wird im Ordner `RafDB-Dataset/` mit den Unterordnern `train/` und `validation/` erwartet, jeweils mit Klassen-Unterordnern (ImageFolder-Format).

## Verwendung

Alle Apps werden vom Projektroot aus gestartet:

### Live-Webcam

```bash
python -m apps.app_webcam              # Standard: alle 2 Sekunden
python -m apps.app_webcam -i 5         # Alle 5 Sekunden klassifizieren
```

Beenden mit **Q**.

### Video-Datei

```bash
python -m apps.app_video mein_video.mp4
python -m apps.app_video mein_video.mp4 -o ergebnis.mp4
```

### Ordner mit Bildern → CSV

```bash
python -m apps.app_folder ./bilder/
python -m apps.app_folder ./bilder/ -o ergebnis.csv
```

Erzeugt eine CSV-Datei mit Softmax-Scores pro Emotion für jedes erkannte Gesicht.

## Erkannte Emotionen

| Klasse     | Label |
|------------|-------|
| Anger      | 0     |
| Disgust    | 1     |
| Fear       | 2     |
| Happiness  | 3     |
| Sadness    | 4     |
| Surprise   | 5     |

## Architektur

Das Modell basiert auf **ResNet-50** (ohne vortrainierte Weights), dessen letzter Fully-Connected-Layer auf 6 Klassen angepasst wird. Training erfolgt mit Adam-Optimizer, CrossEntropyLoss und Cosine-Annealing-Scheduler. Zur Face Detection wird **dlib** (HOG-basiert) verwendet.

## Training anpassen

Die wichtigsten Hyperparameter lassen sich in `src/training.py` beim Aufruf von `run_training()` ändern:

```python
run_training(device, dataset, batch_size=32, learning_rate=0.001, num_epochs=20)
```

Augmentationen (Flip, Crop, ColorJitter) sind in `src/dataset.py` definiert.

## Abhängigkeiten

Siehe `requirements.txt`. Die wichtigsten Pakete:

- `torch` / `torchvision`
- `opencv-python`
- `dlib`
- `Pillow`
- `matplotlib`
- `numpy`
