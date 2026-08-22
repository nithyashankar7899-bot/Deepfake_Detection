# Deepfake Video Detection Using Deep Learning

A deep learning system that classifies videos as real or fake using a
**ResNeXt-50 CNN** (per-frame spatial feature extractor) combined with an
**LSTM** (temporal sequence modelling across frames).

## Architecture

```
Raw video
   │
   ▼
Frame splitting (fixed sampling rate)
   │
   ▼
Face detection (face_recognition / HOG)
   │
   ▼
Cropping + resizing (112x112, with margin)
   │
   ▼
Face-only video reconstruction
   │
   ▼
ResNeXt-50 (ImageNet-pretrained, FC head removed) → 2048-d feature per frame
   │
   ▼
LSTM over the frame-feature sequence
   │
   ▼
Fully connected classifier → REAL / FAKE + confidence score
```

## Project structure

```
CodeAlpha_DeepfakeDetection/
├── data/
│   ├── raw_videos/          # original videos (not included — add your own)
│   │   ├── real/
│   │   └── fake/
│   └── face_videos/         # output of preprocess.py
│       ├── real/
│       └── fake/
├── models/                  # saved checkpoints (best_model.pth)
├── scripts/
│   ├── preprocess.py        # frame splitting, face detection, cropping, reconstruction
│   ├── dataset.py            # PyTorch Dataset for face-crop video sequences
│   ├── model.py               # ResNeXt50 + LSTM architecture
│   ├── train.py                # training loop, checkpointing, validation
│   └── predict.py              # inference on a single video
├── notebooks/                # optional exploratory notebooks
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`face_recognition` depends on `dlib`, which needs CMake and a C++ compiler.
On Ubuntu: `sudo apt install cmake build-essential`. On Windows, installing
via `conda install -c conda-forge dlib` first is usually the smoothest path.

## Datasets

This kind of model is normally trained on public benchmark datasets:

- **FaceForensics++** — https://github.com/ondyari/FaceForensics
- **DFDC (DeepFake Detection Challenge)** — https://ai.meta.com/datasets/dfdc/
- **Celeb-DF** — https://github.com/yuezunli/celeb-deepfakeforensics

Download raw videos and arrange them as:
```
data/raw_videos/real/*.mp4
data/raw_videos/fake/*.mp4
```

## Usage

**1. Preprocess** — extract, detect, crop, and reconstruct face-only videos:
```bash
python scripts/preprocess.py \
    --input_dir data/raw_videos \
    --output_dir data/face_videos \
    --num_frames 100 \
    --img_size 112
```
Note: run this separately for the `real/` and `fake/` subfolders (or adjust
`--input_dir`/`--output_dir` per class) so the output lands in
`data/face_videos/real/` and `data/face_videos/fake/`.

**2. Train**:
```bash
python scripts/train.py \
    --root_dir data/face_videos \
    --epochs 20 \
    --batch_size 4 \
    --seq_len 20 \
    --lr 1e-4 \
    --out_dir models
```
The best checkpoint (by validation accuracy) is saved to `models/best_model.pth`.

**3. Predict on a new video**:
```bash
python scripts/predict.py \
    --video path/to/test_video.mp4 \
    --checkpoint models/best_model.pth
```
Output:
```
Prediction: FAKE  |  Confidence: 91.83%
```

## Notes on reported metrics

- Accuracy and confidence numbers depend heavily on which dataset split you
  train/validate on, sequence length, and number of epochs. Treat any
  specific percentage (e.g. "92% confidence", "88.6% accuracy") as something
  you report **after** you've actually run training and evaluation on your
  chosen dataset — don't state a fixed number until you've reproduced it
  end-to-end, since you may be asked how you got it in an interview.
- `--freeze_cnn` in `train.py` freezes the ResNeXt backbone and only trains
  the LSTM + classifier head — much faster, useful if you have a small
  dataset or limited GPU time.
- For a quick GPU-free sanity check, run `python scripts/model.py` — it
  passes a dummy tensor through the network and prints the output shape.

## Tech stack

Python, PyTorch, torchvision (ResNeXt-50), OpenCV, face_recognition (dlib),
scikit-learn (metrics), pandas.
