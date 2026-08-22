"""
train.py
---------
Training loop for the ResNeXt+LSTM deepfake video classifier.

Run:
    python train.py --root_dir data/face_videos --epochs 20 --batch_size 4 \
                     --seq_len 20 --lr 1e-4 --out_dir models
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

from dataset import DeepfakeVideoDataset
from model import get_model


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, all_preds, all_labels = 0.0, [], []

    for frames, labels in tqdm(loader, desc="Train", leave=False):
        frames, labels = frames.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * frames.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []

    for frames, labels in tqdm(loader, desc="Val", leave=False):
        frames, labels = frames.to(device), labels.to(device)
        logits = model(frames)
        loss = criterion(logits, labels)

        running_loss += loss.item() * frames.size(0)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    return epoch_loss, epoch_acc, cm


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = DeepfakeVideoDataset(
        root_dir=args.root_dir, seq_len=args.seq_len, img_size=args.img_size
    )
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = get_model(pretrained=True, freeze_cnn=args.freeze_cnn).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=2)

    os.makedirs(args.out_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, cm = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        print(f"Confusion matrix:\n{cm}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> new best model saved to {ckpt_path} (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNeXt+LSTM deepfake detector")
    parser.add_argument("--root_dir", required=True, help="Folder with real/ and fake/ subfolders")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=20)
    parser.add_argument("--img_size", type=int, default=112)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--freeze_cnn", action="store_true", help="Freeze ResNeXt backbone")
    parser.add_argument("--out_dir", default="models")
    args = parser.parse_args()

    main(args)
