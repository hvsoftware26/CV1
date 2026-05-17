import os
import json
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from services.video_service import read_video_frames
from services.config import (
    TRAIN_DIR,
    NUM_FRAMES,
    IMG_SIZE,
    BATCH_SIZE,
    SEED,
    LABEL_MAP_PATH,
)


class SignVideoDataset(Dataset):
    def __init__(self, samples, label_to_id):
        self.samples = samples
        self.label_to_id = label_to_id

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        video_path, label = self.samples[index]

        frames = read_video_frames(
            video_path=video_path,
            num_frames=NUM_FRAMES,
            img_size=IMG_SIZE
        )

        label_id = self.label_to_id[label]

        return frames, torch.tensor(label_id, dtype=torch.long)


def build_samples(train_dir=TRAIN_DIR):
    samples = []

    labels = sorted([
        name for name in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, name))
    ])

    for label in labels:
        label_dir = os.path.join(train_dir, label)

        for file_name in os.listdir(label_dir):
            if file_name.lower().endswith(".mp4"):
                video_path = os.path.join(label_dir, file_name)
                samples.append((video_path, label))

    return samples


def build_label_maps(samples):
    labels = sorted(list(set([label for _, label in samples])))

    label_to_id = {
        label: index
        for index, label in enumerate(labels)
    }

    id_to_label = {
        index: label
        for label, index in label_to_id.items()
    }

    return labels, label_to_id, id_to_label


def save_label_map(label_to_id, id_to_label):
    with open(LABEL_MAP_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label_to_id": label_to_id,
                "id_to_label": id_to_label
            },
            f,
            ensure_ascii=False,
            indent=4
        )


def create_class_weights(train_samples, labels, device):
    """
    Tạo class weight vì dataset lệch lớp.
    Class ít video sẽ có trọng số cao hơn.
    """

    train_labels = [label for _, label in train_samples]
    counter = Counter(train_labels)

    weights = []

    for label in labels:
        weights.append(1.0 / counter[label])

    weights = torch.tensor(weights, dtype=torch.float32).to(device)

    return weights


def create_dataloaders(device):
    samples = build_samples(TRAIN_DIR)

    labels, label_to_id, id_to_label = build_label_maps(samples)

    train_samples, val_samples = train_test_split(
        samples,
        test_size=0.2,
        random_state=SEED,
        stratify=[label for _, label in samples]
    )

    train_dataset = SignVideoDataset(train_samples, label_to_id)
    val_dataset = SignVideoDataset(val_samples, label_to_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    class_weights = create_class_weights(
        train_samples=train_samples,
        labels=labels,
        device=device
    )

    save_label_map(label_to_id, id_to_label)

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "labels": labels,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "class_weights": class_weights,
        "num_classes": len(labels),
        "total_samples": len(samples),
        "train_samples": len(train_samples),
        "val_samples": len(val_samples),
    }