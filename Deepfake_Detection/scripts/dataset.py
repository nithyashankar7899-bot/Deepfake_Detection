"""
dataset.py
-----------
PyTorch Dataset for loading preprocessed face-crop videos and returning
a fixed-length sequence of frame tensors, ready for the ResNeXt+LSTM model.

Expected folder layout after preprocess.py:
    data/face_videos/real/*.mp4
    data/face_videos/fake/*.mp4

Or a labels CSV with columns: video_path,label   (label: 0=real, 1=fake)
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def default_transform(img_size=112):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class DeepfakeVideoDataset(Dataset):
    """
    Loads a fixed number of frames per video and stacks them into a
    (seq_len, C, H, W) tensor for the LSTM to consume.
    """

    def __init__(self, csv_path=None, root_dir=None, seq_len=20, img_size=112, transform=None):
        """
        Either supply csv_path (columns: video_path,label) OR root_dir
        with real/ and fake/ subfolders.
        """
        self.seq_len = seq_len
        self.transform = transform or default_transform(img_size)

        if csv_path is not None:
            df = pd.read_csv(csv_path)
            self.samples = list(zip(df["video_path"], df["label"]))
        elif root_dir is not None:
            self.samples = []
            for label, cls in enumerate(["real", "fake"]):
                cls_dir = os.path.join(root_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".mp4", ".avi")):
                        self.samples.append((os.path.join(cls_dir, fname), label))
        else:
            raise ValueError("Provide either csv_path or root_dir")

    def __len__(self):
        return len(self.samples)

    def _load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened() and len(frames) < self.seq_len:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()

        if len(frames) == 0:
            # fallback: black frames if video failed to load
            frames = [np.zeros((112, 112, 3), dtype=np.uint8)]

        # pad short sequences by repeating the last frame
        while len(frames) < self.seq_len:
            frames.append(frames[-1])

        return frames[: self.seq_len]

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        frames = self._load_frames(video_path)
        tensor_frames = torch.stack([self.transform(f) for f in frames])  # (seq_len, C, H, W)
        return tensor_frames, torch.tensor(label, dtype=torch.long)
