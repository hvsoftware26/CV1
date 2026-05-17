import os
import json

import torch
from tqdm import tqdm

from services.config import (
    TEST_DIR,
    MODEL_PATH,
    LABEL_MAP_PATH,
    NUM_FRAMES,
    IMG_SIZE,
)
from services.video_service import read_video_frames
from services.model_service import build_model


def load_id_to_label():
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    id_to_label = {
        int(key): value
        for key, value in label_data["id_to_label"].items()
    }

    return id_to_label


def predict_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    id_to_label = load_id_to_label()

    num_classes = len(id_to_label)

    model = build_model(
        num_classes=num_classes,
        device=device
    )

    model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device)
    )

    model.eval()

    video_files = sorted([
        file_name for file_name in os.listdir(TEST_DIR)
        if file_name.lower().endswith(".mp4")
    ])

    results = []

    with torch.no_grad():
        for video_name in tqdm(video_files, desc="Predicting"):
            video_path = os.path.join(TEST_DIR, video_name)

            frames = read_video_frames(
                video_path=video_path,
                num_frames=NUM_FRAMES,
                img_size=IMG_SIZE
            )

            frames = frames.unsqueeze(0).to(device)

            outputs = model(frames)

            pred_id = torch.argmax(outputs, dim=1).item()

            pred_label = id_to_label[pred_id]

            results.append({
                "video_name": video_name,
                "label": pred_label
            })

    return results