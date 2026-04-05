import torch
import torch.nn as nn

from dataset import VizDoomDataset
from model import VizDoomCNN
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dataset import VizDoomDataset


def resolve_dataset_path(file_name):
    """Resolve dataset path across project layouts"""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    candidates = [
        project_root / "data" / "raw" / file_name,
        script_dir / "data" / "raw" / file_name,
        Path.cwd() / "data" / "raw" / file_name,
        project_root / "Deep-Learning" / "data" / "raw" / file_name,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Could not find '{file_name}'. Checked: {candidates}")

dataset_path = resolve_dataset_path("novice_dataset.npz")
dataset = VizDoomDataset(dataset_path)

frames = torch.stack([dataset[i][0] for i in range(8)])
actions = torch.stack([dataset[i][1] for i in range(8)])

print("Frames shape:", frames.shape)
print("Actions shape:", actions.shape)

model = VizDoomCNN()

button_logits, mouse_output = model(frames)

print("Button logits shape:", button_logits.shape)
print("Mouse output shape:", mouse_output.shape)

buttons = actions[:, :5]
mouse = actions[:, 5:]

button_criterion = nn.BCEWithLogitsLoss()
mouse_criterion = nn.MSELoss()

button_loss = button_criterion(button_logits, buttons)
mouse_loss = mouse_criterion(mouse_output, mouse)
total_loss = button_loss + mouse_loss

print("Button loss:", button_loss.item())
print("Mouse loss:", mouse_loss.item())
print("Total loss:", total_loss.item())