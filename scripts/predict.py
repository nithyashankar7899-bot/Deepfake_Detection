"""
predict.py
-----------
Run inference on a single video (or a raw, un-preprocessed video) and
output a real/fake label with a confidence score.

Run:
    python predict.py --video path/to/video.mp4 --checkpoint models/best_model.pth
"""

import os
import argparse
import cv2
import torch
import torch.nn.functional as F

from model import get_model
from dataset import default_transform


LABELS = {0: "REAL", 1: "FAKE"}

# Built into OpenCV — no extra download or compiling needed.
_CASCADE_PATH = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
_FACE_DETECTOR = cv2.CascadeClassifier(_CASCADE_PATH)


def extract_face_sequence(video_path, seq_len=20, img_size=112, margin=0.3):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total // (seq_len * 3))  # oversample, since some frames may have no face

    faces = []
    frame_idx = 0
    while cap.isOpened() and len(faces) < seq_len:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = _FACE_DETECTOR.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
            )
            if len(detections) > 0:
                x, y, box_w, box_h = max(detections, key=lambda b: b[2] * b[3])
                h, w = frame.shape[:2]
                pad_h, pad_w = int(box_h * margin), int(box_w * margin)
                top, bottom = max(0, y - pad_h), min(h, y + box_h + pad_h)
                left, right = max(0, x - pad_w), min(w, x + box_w + pad_w)
                face = frame[top:bottom, left:right]
                face = cv2.cvtColor(cv2.resize(face, (img_size, img_size)), cv2.COLOR_BGR2RGB)
                faces.append(face)
        frame_idx += 1
    cap.release()

    if not faces:
        raise RuntimeError("No face detected in this video.")

    while len(faces) < seq_len:
        faces.append(faces[-1])

    return faces[:seq_len]


def predict(video_path, checkpoint_path, seq_len=20, img_size=112, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = get_model(pretrained=False)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    transform = default_transform(img_size)
    frames = extract_face_sequence(video_path, seq_len, img_size)
    tensor = torch.stack([transform(f) for f in frames]).unsqueeze(0).to(device)  # (1, seq_len, C, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item()) * 100

    return LABELS[pred_idx], confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run deepfake detection on a video")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model .pth file")
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=112)
    args = parser.parse_args()

    label, confidence = predict(args.video, args.checkpoint, args.seq_len, args.img_size)
    print(f"Prediction: {label}  |  Confidence: {confidence:.2f}%")
