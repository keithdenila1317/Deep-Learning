import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def apply_center_bottom_focus(frame, center_weight=0.35, bottom_weight=0.35):
    """Bias a grayscale frame toward the screen center and bottom-center.

    This keeps the original frame content but blends in resized crops from the
    center and the lower middle of the image, which is useful for aiming and
    weapon visibility in first-person VizDoom frames.
    """
    if not torch.is_tensor(frame):
        frame = torch.tensor(frame, dtype=torch.float32)

    if frame.dim() == 2:
        frame = frame.unsqueeze(0).unsqueeze(0)
    elif frame.dim() == 3:
        frame = frame.unsqueeze(0)

    _, _, height, width = frame.shape

    center_crop_h = max(8, min(height, int(height * 0.60)))
    center_crop_w = max(8, min(width, int(width * 0.60)))
    center_top = max(0, (height - center_crop_h) // 2)
    center_left = max(0, (width - center_crop_w) // 2)

    bottom_crop_h = max(8, min(height, int(height * 0.45)))
    bottom_crop_w = max(8, min(width, int(width * 0.60)))
    bottom_top = max(0, height - bottom_crop_h)
    bottom_left = max(0, (width - bottom_crop_w) // 2)

    center_crop = frame[:, :, center_top:center_top + center_crop_h, center_left:center_left + center_crop_w]
    bottom_crop = frame[:, :, bottom_top:bottom_top + bottom_crop_h, bottom_left:bottom_left + bottom_crop_w]

    center_crop = F.interpolate(center_crop, size=(height, width), mode="bilinear", align_corners=False)
    bottom_crop = F.interpolate(bottom_crop, size=(height, width), mode="bilinear", align_corners=False)

    base_weight = max(0.0, 1.0 - center_weight - bottom_weight)
    enhanced = base_weight * frame + center_weight * center_crop + bottom_weight * bottom_crop

    return enhanced.squeeze(0)


class VizDoomDataset(Dataset):
    def __init__(self, npz_path, sequence_length=1, focus_center_bottom=True):
        data = np.load(npz_path)

        self.frames = data["frames"]
        self.actions = data["actions"]
        self.sequence_length = max(1, int(sequence_length))
        self.focus_center_bottom = focus_center_bottom

    def __len__(self):
        return max(0, len(self.frames) - self.sequence_length + 1)

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        frame_sequence = self.frames[idx:end_idx]
        action = self.actions[end_idx - 1]

        # normalize image sequence
        frame_sequence = frame_sequence.astype(np.float32) / 255.0

        if self.focus_center_bottom:
            focused_frames = []
            for frame in frame_sequence:
                focused_frames.append(apply_center_bottom_focus(frame))
            frame_sequence = torch.stack(focused_frames, dim=0).squeeze(1).numpy()

        # add channel dimension -> [T, 1, H, W]
        frame_sequence = np.expand_dims(frame_sequence, axis=1)

        frame = torch.tensor(frame_sequence, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.float32)

        return frame, action


class SequentialVizDoomDataset(Dataset):
    """Dataset that returns temporal sequences for training with LSTM.
    
    Returns:
        frames: [T, 1, H, W] - sequence of T frames
        buttons: [5] - binary button press (ground truth action)
        mouse: [2] - normalized mouse movement (ground truth action)
    """
    def __init__(self, npz_path, sequence_length=4, is_train=True, focus_center_bottom=True):
        data = np.load(npz_path)
        self.frames = data["frames"]
        
        # Split action into buttons and mouse
        self.actions = data["actions"]  # [N, 7]: 5 buttons + 2 mouse
        self.sequence_length = max(1, int(sequence_length))
        self.is_train = is_train
        self.focus_center_bottom = focus_center_bottom

    def __len__(self):
        return max(0, len(self.frames) - self.sequence_length + 1)

    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        frame_sequence = self.frames[idx:end_idx]
        action_target = self.actions[end_idx - 1]  # Predict action at end of sequence

        # Normalize frames
        frame_sequence = frame_sequence.astype(np.float32) / 255.0

        if self.focus_center_bottom:
            focused_frames = []
            for frame in frame_sequence:
                focused_frames.append(apply_center_bottom_focus(frame))
            frame_sequence = torch.stack(focused_frames, dim=0).squeeze(1).numpy()
        
        # Add channel dimension -> [T, 1, H, W]
        frame_sequence = np.expand_dims(frame_sequence, axis=1)

        frames = torch.tensor(frame_sequence, dtype=torch.float32)
        
        # Split action into buttons and mouse
        buttons = torch.tensor(action_target[:5], dtype=torch.float32)
        mouse = torch.tensor(action_target[5:] / 55.0, dtype=torch.float32)

        return frames, buttons, mouse