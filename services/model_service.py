import torch
import torch.nn as nn


class SmallCNN_GRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Input x shape:
            [B, T, C, H, W]

        Ví dụ:
            [2, 30, 3, 128, 128]
        """

        batch_size, time_steps, c, h, w = x.shape

        x = x.view(batch_size * time_steps, c, h, w)

        features = self.cnn(x)

        features = features.view(batch_size, time_steps, 128)

        output, hidden = self.gru(features)

        last_hidden = hidden[-1]

        logits = self.classifier(last_hidden)

        return logits


def build_model(num_classes, device):
    model = SmallCNN_GRU(num_classes=num_classes)
    model = model.to(device)

    return model