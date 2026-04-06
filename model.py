import torch
import torch.nn as nn


class VizDoomCNN(nn.Module):
    def __init__(self, dropout=0.3, hidden_size=256, lstm_layers=1):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.flatten = nn.Flatten()

        self.frame_encoder = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            nn.Linear(512, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.temporal = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
        )

        self.button_head = nn.Linear(hidden_size, 5)
        self.mouse_head = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # Accept both [B, C, H, W] and [B, T, C, H, W] for compatibility
        if x.dim() == 4:
            x = x.unsqueeze(1)

        batch_size, seq_len, channels, height, width = x.shape

        x = x.reshape(batch_size * seq_len, channels, height, width)
        x = self.features(x)
        x = self.flatten(x)
        x = self.frame_encoder(x)
        x = x.reshape(batch_size, seq_len, -1)

        temporal_out, _ = self.temporal(x)
        x = temporal_out[:, -1, :]

        button_logits = self.button_head(x)
        mouse_output = self.mouse_head(x)

        return button_logits, mouse_output

