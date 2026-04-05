import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import VizDoomDataset
from model import VizDoomCNN


def resolve_dataset_path(dataset_path):
    """Resolve dataset path from cwd, src/, or project root layouts."""
    raw_path = Path(dataset_path)
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent

    candidates = [
        raw_path,
        Path.cwd() / raw_path,
        script_dir / raw_path,
        project_root / raw_path,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    checked = [str(path) for path in candidates]
    raise FileNotFoundError(
        f"Could not find dataset at '{dataset_path}'. Checked: {checked}"
    )


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


def compute_button_pos_weight(loader, num_buttons=5, eps=1e-6, clamp_min=0.1, clamp_max=20.0):
    """Compute per-button positive weights as neg/pos for BCEWithLogitsLoss."""
    pos_count = torch.zeros(num_buttons, dtype=torch.float64)
    total_count = 0

    for _, actions in loader:
        buttons = actions[:, :num_buttons]
        pos_count += buttons.sum(dim=0).to(torch.float64)
        total_count += buttons.size(0)

    if total_count == 0:
        return torch.ones(num_buttons, dtype=torch.float32)

    total_per_button = torch.full_like(pos_count, float(total_count))
    neg_count = total_per_button - pos_count
    pos_weight = neg_count / torch.clamp(pos_count, min=eps)
    pos_weight = torch.clamp(pos_weight, min=clamp_min, max=clamp_max)

    return pos_weight.to(torch.float32)


def train_one_epoch(model, loader, optimizer, button_criterion, mouse_criterion, device):
    model.train()
    running_loss = 0.0

    for frames, actions in tqdm(loader, desc="Training", leave=False):
        frames = frames.to(device)
        actions = actions.to(device)

        buttons = actions[:, :5]
        mouse = actions[:, 5:]

        optimizer.zero_grad()

        button_logits, mouse_pred = model(frames)

        button_loss = button_criterion(button_logits, buttons)
        mouse_loss = mouse_criterion(mouse_pred, mouse)
        loss = button_loss + mouse_loss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate_one_epoch(model, loader, button_criterion, mouse_criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for frames, actions in tqdm(loader, desc="Validation", leave=False):
            frames = frames.to(device)
            actions = actions.to(device)

            buttons = actions[:, :5]
            mouse = actions[:, 5:]

            button_logits, mouse_pred = model(frames)

            button_loss = button_criterion(button_logits, buttons)
            mouse_loss = mouse_criterion(mouse_pred, mouse)
            loss = button_loss + mouse_loss

            running_loss += loss.item()

    return running_loss / len(loader)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = resolve_dataset_path(args.dataset_path)
    print(f"Using dataset: {dataset_path}")
    dataset = VizDoomDataset(dataset_path, sequence_length=args.sequence_length)

    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed)

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    print(f"Total samples: {len(dataset)}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = VizDoomCNN(dropout=args.dropout).to(device)

    if not args.no_weighted_bce:
        pos_weight = compute_button_pos_weight(
            train_loader,
            num_buttons=5,
            clamp_min=args.pos_weight_min,
            clamp_max=args.pos_weight_max,
        ).to(device)
        button_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using weighted BCE with pos_weight={pos_weight.tolist()}")
    else:
        button_criterion = nn.BCEWithLogitsLoss()
        print("Using standard BCEWithLogitsLoss")

    if not args.no_signed_mouse_loss:
        mouse_criterion = SignedMSELoss(wrong_sign_scale=args.wrong_sign_scale)
        print(f"Using SignedMSELoss with wrong_sign_scale={args.wrong_sign_scale}")
    else:
        mouse_criterion = nn.MSELoss()
        print("Using standard MSELoss")

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    os.makedirs(args.save_dir, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
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

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")

        last_checkpoint_path = os.path.join(args.save_dir, f"{args.save_name}_last.pth")
        torch.save(model.state_dict(), last_checkpoint_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = os.path.join(args.save_dir, f"{args.save_name}_best.pth")
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Saved best model to {best_checkpoint_path}")

    print("Training complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to .npz dataset file"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="vizdoom_model",
        help="Base name for saved model files"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate"
    )
    parser.add_argument(
        "--train_split",
        type=float,
        default=0.8,
        help="Fraction of data used for training"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout rate for the model"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of DataLoader workers"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split"
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=8,
        help="Number of consecutive frames per training sample"
    )
    parser.add_argument(
        "--no_weighted_bce",
        action="store_true",
        help="Disable per-button class weights for BCE loss"
    )
    parser.add_argument(
        "--pos_weight_min",
        type=float,
        default=0.1,
        help="Minimum clamp value for button pos_weight"
    )
    parser.add_argument(
        "--pos_weight_max",
        type=float,
        default=20.0,
        help="Maximum clamp value for button pos_weight"
    )
    parser.add_argument(
        "--no_signed_mouse_loss",
        action="store_true",
        help="Disable signed-aware MSE for mouse prediction"
    )
    parser.add_argument(
        "--wrong_sign_scale",
        type=float,
        default=0.33,
        help="Penalty scale for wrong mouse sign in SignedMSELoss"
    )

    args = parser.parse_args()
    main(args)