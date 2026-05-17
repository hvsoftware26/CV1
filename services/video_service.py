import cv2
import numpy as np
import torch


def preprocess_frame(frame, img_size):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (img_size, img_size))
    return frame

def read_video_frames(video_path, num_frames=30, img_size=128):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return torch.zeros(num_frames, 3, img_size, img_size)
    frame_indices = np.linspace(0, total_frames - 1, num_frames)
    frame_indices = np.round(frame_indices).astype(int)
    target_indices = set(frame_indices)

    frames = []
    current_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if current_index in target_indices:
            frame = preprocess_frame(frame, img_size)
            frame = frame.astype(np.float32) / 255.0
            frame = torch.tensor(frame).permute(2, 0, 1)
            frames.append(frame)

        current_index += 1

    cap.release()

    if len(frames) == 0:
        frames = [torch.zeros(3, img_size, img_size)]

    while len(frames) < num_frames:
        frames.append(frames[-1])

    frames = frames[:num_frames]
    frames = torch.stack(frames)
    return frames