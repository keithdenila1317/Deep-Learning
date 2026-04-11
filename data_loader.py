import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import torchvision.transforms as transformers
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from dataset import SequentialVizDoomDataset, apply_center_bottom_focus


class Augmentation :
    
    # change is_train when validating with testing
    def __init__ (self, dataset_npz, is_train = True, focus_center_bottom = True) :

        # ram issue change np.load to load batches instead of entire array
        data = np.load(dataset_npz)
        self.frames = data['frames']
        self.actions = data['actions']
        self.is_train = is_train
        self.focus_center_bottom = focus_center_bottom

        # adds noise / differientiates frames 
        if self.is_train :

            self.transform = transformers.Compose([transformers.ColorJitter(brightness = 0.2, contrast = 0.2)])
        else :
            self.transform = None

    def __len__ (self) :
        
        return len(self.frames)
    
    def __getitem__ (self, index) :

        # normalization with correct channel dimension (x, x, x)
        frame = self.frames[index].astype(np.float32) / 255.0
        frame = torch.tensor(frame).unsqueeze(0)

        # training goes through transformation, will skip when testing
        if self.transform :

            frame = self.transform(frame)

        if self.focus_center_bottom:
            frame = apply_center_bottom_focus(frame)

        action = self.actions[index]

        # Movement (W, A, S, D, and Mouse 1)
        binary_actions = torch.tensor(action[:5], dtype = torch.float32)

        # Vertical and Horizontal mouse movement ([up,down], [left,right])
        normalized_actions = action[5:] / 55.0
        float_actions = torch.tensor(normalized_actions, dtype = torch.float32)

        return frame, binary_actions, float_actions
    
def get_loaders(dataset_npz, batch_size=32, sequence_length=1, focus_center_bottom=True):
    """Return train/val loaders for either single-frame or temporal training.

    sequence_length=1 -> single-frame loader (legacy behavior).
    sequence_length>1 -> temporal sequence loader [B, T, C, H, W].
    """
    if sequence_length > 1:
        dataset = SequentialVizDoomDataset(
            dataset_npz,
            sequence_length=sequence_length,
            is_train=True,
            focus_center_bottom=focus_center_bottom,
        )
        train_size = int(0.80 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        return train_loader, val_loader

    # train/test switch (single-frame)
    data_set = Augmentation(dataset_npz, is_train=True, focus_center_bottom=focus_center_bottom)

    # split 80-20 can be adjusted
    training_split = int(0.80 * len(data_set))
    validation_split = len(data_set) - training_split

    # splitting the frames into the two sections randomly
    training_split, validation_split = random_split(data_set, [training_split, validation_split])

    # when testing, disable augmenting
    validation_split.dataset.is_train = False

    training_loader = DataLoader(training_split, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_split, batch_size=batch_size, shuffle=False)

    return training_loader, validation_loader


def get_sequential_loaders(dataset_npz, batch_size=32, sequence_length=4, focus_center_bottom=True):
    """Compatibility wrapper for temporal training."""
    return get_loaders(
        dataset_npz,
        batch_size=batch_size,
        sequence_length=sequence_length,
        focus_center_bottom=focus_center_bottom,
    )

