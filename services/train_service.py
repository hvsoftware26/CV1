
import os
import random
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

from services.config import EPOCHS, LR, MODEL_PATH, SEED
from services.data_service import create_dataloaders
from services.model_service import build_model


OLD_BEST_F1 = 0.2657


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_best_model_if_exists(model, device):

    if os.path.exists(MODEL_PATH):
        print("Found existing best_model.pth.")
        print("Loading weights from:", MODEL_PATH)

        model.load_state_dict(
            torch.load(MODEL_PATH, map_location=device)
        )
        return True

    print("No existing best_model.pth found")
    return False


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()

    total_loss = 0.0

    for frames, targets in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
        frames = frames.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(frames)

        loss = criterion(outputs, targets)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    return avg_loss


def evaluate(model, val_loader, criterion, device, labels, epoch):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for frames, targets in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
            frames = frames.to(device)
            targets = targets.to(device)

            outputs = model(frames)

            loss = criterion(outputs, targets)

            preds = torch.argmax(outputs, dim=1)

            total_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(val_loader)

    macro_f1 = f1_score(
        all_targets,
        all_preds,
        average="macro"
    )

    report = classification_report(
        all_targets,
        all_preds,
        target_names=labels,
        zero_division=0
    )

    return avg_loss, macro_f1, report


def run_training():
    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    data = create_dataloaders(device)

    print("Số class:", data["num_classes"])
    print("Tổng samples:", data["total_samples"])
    print("Train samples:", data["train_samples"])
    print("Val samples:", data["val_samples"])

    model = build_model(
        num_classes=data["num_classes"],
        device=device
    )

    loaded_old_model = load_best_model_if_exists(
        model=model,
        device=device
    )

    criterion = nn.CrossEntropyLoss(
        weight=data["class_weights"]
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    if loaded_old_model:
        best_f1 = OLD_BEST_F1
        print("Continue training from old best model.")
        print("Initial best_f1:", best_f1)
    else:
        best_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model=model,
            train_loader=data["train_loader"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch
        )

        val_loss, macro_f1, report = evaluate(
            model=model,
            val_loader=data["val_loader"],
            criterion=criterion,
            device=device,
            labels=data["labels"],
            epoch=epoch
        )

        print()
        print("Epoch:", epoch)
        print("Train loss:", round(train_loss, 4))
        print("Val loss:", round(val_loss, 4))
        print("Val Macro-F1:", round(macro_f1, 4))
        print(report)

        if macro_f1 > best_f1:
            best_f1 = macro_f1

            torch.save(model.state_dict(), MODEL_PATH)

            print("Saved best model:", round(best_f1, 4))

    print("Training finished.")
    print("Best Macro-F1:", best_f1)