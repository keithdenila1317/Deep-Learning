
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import os
from data_loader import get_loaders
from model import VizDoomCNN

class SignedMSELoss(nn.Module):
    """MSE that penalizes wrong-sign mouse predictions more heavily."""

    def __init__(self, wrong_sign_scale=0.33, zero_target_eps=1e-6):
        super().__init__()
        self.wrong_sign_scale = wrong_sign_scale
        self.zero_target_eps = zero_target_eps

    def forward(self, prediction, target):
        squared_error = (prediction - target) ** 2

        same_sign = (prediction * target) > 0
        near_zero_target = torch.abs(target) <= self.zero_target_eps

        sign_mask = torch.where(
            near_zero_target,
            torch.ones_like(target),
            torch.where(
                same_sign,
                torch.ones_like(target),
                torch.full_like(target, self.wrong_sign_scale),
            ),
        )

        return (squared_error / sign_mask).mean()

def train_one_epoch(model, loader, optimizer, button_criterion, mouse_criterion, device):
    model.train()
    running_loss = 0.0

    for frames, buttons, mouse in tqdm(loader, desc="Training", leave=False):
        frames = frames.to(device)
        buttons = buttons.to(device)
        mouse = mouse.to(device)

        optimizer.zero_grad()

        button_logits, mouse_pred = model(frames)

        button_loss = button_criterion(button_logits, buttons)
        mouse_loss = mouse_criterion(mouse_pred, mouse)
        loss = button_loss + (mouse_loss * 5.0)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, button_criterion, mouse_criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for frames, buttons, mouse in tqdm(loader, desc="Validation", leave=False):
            frames = frames.to(device)
            buttons = buttons.to(device)
            mouse = mouse.to(device)

            button_logits, mouse_pred = model(frames)

            button_loss = button_criterion(button_logits, buttons)
            mouse_loss = mouse_criterion(mouse_pred, mouse)
            loss = button_loss + mouse_loss

            running_loss += loss.item()

    return running_loss / len(loader)


def main():

    dataset_path = "novice_dataset.npz"
    epochs = 50
    batch_size = 32
    learning_rate = 1e-3
    save_dir = "."
    save_name = "novice_model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_loaders(dataset_path, batch_size = batch_size, seq_len = 15)
    model = VizDoomCNN(dropout = 0.3).to(device)



    novice_weights = torch.tensor([5.0, 1.0, 1.0, 1.0, 2.0], dtype = torch.float32).to(device)
    button_criterion = nn.BCEWithLogitsLoss(pos_weight = novice_weights)
    #mouse_criterion = SignedMSELoss(wrong_sign_scale = 0.33)

    #button_criterion = nn.BCEWithLogitsLoss() 
    mouse_criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    os.makedirs(save_dir, exist_ok = True)
    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            button_criterion,
            mouse_criterion,
            device
        )

        val_loss = validate_one_epoch(
            model,
            val_loader,
            button_criterion,
            mouse_criterion,
            device
        )

        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")

        last_checkpoint_path = os.path.join(save_dir, f"{save_name}_last.pth")
        torch.save(model.state_dict(), last_checkpoint_path)

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(save_dir, f"{save_name}_best.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"  -> Saved new best model to {best_checkpoint_path}")

        print("Complete")

if __name__ == "__main__":
    main()