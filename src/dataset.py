import numpy as np
import torch
from torch.utils.data import Dataset


class VizDoomDataset(Dataset):
    def __init__(self, npz_path, sequence_length=1):
        data = np.load(npz_path)

        self.frames = data["frames"]
        self.actions = data["actions"]
        self.sequence_length = max(1, int(sequence_length))

    def __len__(self):
        return max(0, len(self.frames) - self.sequence_length + 1)

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        frame_sequence = self.frames[idx:end_idx]
        action = self.actions[end_idx - 1]

        # normalize image sequence
        frame_sequence = frame_sequence.astype(np.float32) / 255.0

        # add channel dimension -> [T, 1, H, W]
        frame_sequence = np.expand_dims(frame_sequence, axis=1)

        frame = torch.tensor(frame_sequence, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)

        return frame, action