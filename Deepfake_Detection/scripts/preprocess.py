"""
preprocess.py
--------------
Preprocessing pipeline for the Deepfake Video Detection project.

Pipeline stages:
    1. Frame splitting  - read every video and pull frames at a fixed rate
    2. Face detection    - locate the face in each frame (face_recognition / HOG+CNN)
    3. Cropping           - crop tightly around the detected face with a small margin
    4. Video reconstruction - re-assemble the cropped face frames into a fixed-length
                              face-only video (this is what the model actually trains on)

Run:
    python preprocess.py --input_dir data/raw_videos --output_dir data/face_videos \
                          --num_frames 100 --img_size 112
"""

import os
import cv2
import glob
import argparse
import face_recognition
from tqdm import tqdm


def extract_frames(video_path, every_n=1):
    """Yield frames from a video, one every `every_n` frames."""
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % every_n == 0:
            yield frame
        frame_idx += 1
    cap.release()


def crop_face(frame, margin=0.3):
    """
    Detect the largest face in a frame and return a square crop around it
    with a margin. Returns None if no face is found.
    """
    # face_recognition expects RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb, model="hog")

    if not face_locations:
        return None

    # pick the largest face box if multiple people are in frame
    def box_area(box):
        top, right, bottom, left = box
        return (bottom - top) * (right - left)

    top, right, bottom, left = max(face_locations, key=box_area)

    h, w = frame.shape[:2]
    box_h, box_w = bottom - top, right - left
    pad_h, pad_w = int(box_h * margin), int(box_w * margin)

    top = max(0, top - pad_h)
    bottom = min(h, bottom + pad_h)
    left = max(0, left - pad_w)
    right = min(w, right + pad_w)

    return frame[top:bottom, left:right]


def process_video(video_path, out_path, num_frames=100, img_size=112):
    """
    Extract `num_frames` face crops from a video, resize them, and write
    them out as a reconstructed face-only video (mp4).
    Returns True on success, False if too few valid faces were found.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total <= 0:
        return False

    step = max(1, total // num_frames)
    face_frames = []

    for frame in extract_frames(video_path, every_n=step):
        face = crop_face(frame)
        if face is None or face.size == 0:
            continue
        face = cv2.resize(face, (img_size, img_size))
        face_frames.append(face)
        if len(face_frames) >= num_frames:
            break

    # require a reasonable fraction of frames to have a detected face,
    # otherwise skip the video (common with heavy occlusion / bad source clips)
    if len(face_frames) < num_frames * 0.5:
        return False

    # pad by repeating the last frame if we came in short
    while len(face_frames) < num_frames:
        face_frames.append(face_frames[-1])

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, 25, (img_size, img_size))
    for f in face_frames:
        writer.write(f)
    writer.release()
    return True


def run(input_dir, output_dir, num_frames, img_size):
    os.makedirs(output_dir, exist_ok=True)
    video_paths = glob.glob(os.path.join(input_dir, "**", "*.mp4"), recursive=True)

    ok, skipped = 0, 0
    for video_path in tqdm(video_paths, desc="Preprocessing videos"):
        rel_name = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(output_dir, f"{rel_name}_faces.mp4")
        try:
            success = process_video(video_path, out_path, num_frames, img_size)
        except Exception as e:
            print(f"[WARN] failed on {video_path}: {e}")
            success = False

        if success:
            ok += 1
        else:
            skipped += 1

    print(f"Done. {ok} videos processed, {skipped} skipped (no reliable face detected).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face extraction preprocessing pipeline")
    parser.add_argument("--input_dir", required=True, help="Folder with raw .mp4 videos")
    parser.add_argument("--output_dir", required=True, help="Folder to save face-cropped videos")
    parser.add_argument("--num_frames", type=int, default=100, help="Frames to sample per video")
    parser.add_argument("--img_size", type=int, default=112, help="Output face crop size (square)")
    args = parser.parse_args()

    run(args.input_dir, args.output_dir, args.num_frames, args.img_size)
