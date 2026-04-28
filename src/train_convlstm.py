import os
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset import VizDoomDataset


def resolve_dataset_path(dataset_path):
    
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


class ConvLSTMCell(nn.Module):
    """ConvLSTM that applies convolution in the recurrent connections."""
    def __init__(self, input_channels, hidden_channels, kernel_size=3, padding=1):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = padding

        # single conv for all 4 gates (i, f, g, o)
        self.conv = nn.Conv2d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        """
        Args:
            x: [B, C_in, H, W]
            h: [B, C_hidden, H, W]
            c: [B, C_hidden, H, W]
        Returns:
            h_new: [B, C_hidden, H, W]
            c_new: [B, C_hidden, H, W]
        """
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)

        # Split into 4 gates
        i, f, g, o = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)

        return h_new, c_new


class ConvLSTM(nn.Module):
    """Multi-layer ConvLSTM."""
    def __init__(self, input_channels, hidden_channels, num_layers=1, kernel_size=3):
        super().__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels if isinstance(hidden_channels, list) else [hidden_channels] * num_layers
        self.num_layers = num_layers
        self.kernel_size = kernel_size

        self.layers = nn.ModuleList([
            ConvLSTMCell(
                input_channels if i == 0 else self.hidden_channels[i-1],
                self.hidden_channels[i],
                kernel_size=kernel_size
            )
            for i in range(num_layers)
        ])

    def forward(self, x, h=None, c=None):
        """
        Args:
            x: [B, T, C, H, W]
            h: list of hidden states or None
            c: list of cell states or None
        Returns:
            output: [B, T, C_hidden, H, W]
            h: list of final hidden states
            c: list of final cell states
        """
        batch_size, seq_len, _, height, width = x.shape

        if h is None:
            h = [torch.zeros(batch_size, self.hidden_channels[i], height, width, device=x.device) 
                 for i in range(self.num_layers)]
        if c is None:
            c = [torch.zeros(batch_size, self.hidden_channels[i], height, width, device=x.device) 
                 for i in range(self.num_layers)]

        output_sequence = []

        for t in range(seq_len):
            x_t = x[:, t, :, :, :]  # [B, C, H, W]

            for layer_idx in range(self.num_layers):
                h[layer_idx], c[layer_idx] = self.layers[layer_idx](x_t, h[layer_idx], c[layer_idx])
                x_t = h[layer_idx]

            output_sequence.append(x_t.unsqueeze(1))

        output = torch.cat(output_sequence, dim=1)  # [B, T, C_hidden, H, W]

        return output, h, c


class VizDoomConvLSTM(nn.Module):
    def __init__(self, dropout=0.3, hidden_channels=64, num_layers=2):
        super().__init__()

        self.initial_conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.convlstm = ConvLSTM(
            input_channels=32,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            kernel_size=3
        )

        self.final_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.flatten = nn.Flatten()

        flattened_size = hidden_channels * 4 * 4
        self.button_head = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 5)
        )

        self.mouse_head = nn.Sequential(
            nn.Linear(flattened_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] or [B, T, C, H, W]
        Returns:
            button_logits: [B, 5]
            mouse_output: [B, 2]
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # [B, 1, C, H, W]

        batch_size, seq_len, _, height, width = x.shape

        x = x.reshape(batch_size * seq_len, 1, height, width)
        x = self.initial_conv(x)
        x = x.reshape(batch_size, seq_len, 32, height, width)

        convlstm_out, _, _ = self.convlstm(x)  # [B, T, hidden_channels, H, W]

        final_frame = convlstm_out[:, -1, :, :, :]  # [B, hidden_channels, H, W]

        pooled = self.final_pool(final_frame)
        flattened = self.flatten(pooled)

        button_logits = self.button_head(flattened)
        mouse_output = self.mouse_head(flattened)

        return button_logits, mouse_output


class SignedMSELoss(nn.Module):
    """MSE that penalizes wrong-sign mouse predictions more heavily."""

    def __init__(self, wrong_sign_scale=0.33, zero_target_eps=1e-6):
        super().__init__()
        self.wrong_sign_scale = wrong_sign_scale
        self.zero_target_eps = zero_target_eps

    def forward(self, pred, target):
        diff = pred - target

        # Check for sign mismatch when target is not zero
        abs_target = torch.abs(target)
        has_magnitude = abs_target > self.zero_target_eps
        sign_match = (pred * target) > 0

        wrong_sign_mask = has_magnitude & ~sign_match

        loss = diff ** 2
        loss[wrong_sign_mask] *= self.wrong_sign_scale

        return loss.mean()


def train_epoch(model, train_loader, optimizer, device, button_criterion, mouse_criterion):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc="Training")

    for frames, buttons, mouse in pbar:
        frames = frames.to(device)
        buttons = buttons.to(device)
        mouse = mouse.to(device)

        optimizer.zero_grad()

        button_logits, mouse_output = model(frames)

        button_loss = button_criterion(button_logits, buttons)
        mouse_loss = mouse_criterion(mouse_output, mouse)
        loss = button_loss + mouse_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({"Loss": loss.item()})

    return total_loss / len(train_loader)


def validate(model, val_loader, device, button_criterion, mouse_criterion):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for frames, buttons, mouse in pbar:
            frames = frames.to(device)
            buttons = buttons.to(device)
            mouse = mouse.to(device)

            button_logits, mouse_output = model(frames)

            button_loss = button_criterion(button_logits, buttons)
            mouse_loss = mouse_criterion(mouse_output, mouse)
            loss = button_loss + mouse_loss

            total_loss += loss.item()
            pbar.set_postfix({"Val Loss": loss.item()})

    return total_loss / len(val_loader)


def main():
    parser = argparse.ArgumentParser(description="Train a ConvLSTM model for one VizDoom dataset")
    parser.add_argument("--dataset_type", type=str, choices=["expert", "novice"], required=True)
    parser.add_argument("--dataset_path", type=str, default="", help="Optional override path to a .npz dataset file")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--convlstm_layers", type=int, default=2)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/v3_convlstm")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--sequence_length", type=int, default=4, help="Number of consecutive frames per sample")

    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.dataset_path:
        dataset_path = resolve_dataset_path(args.dataset_path)
    else:
        default_path = "data/raw/expert_dataset.npz" if args.dataset_type == "expert" else "data/raw/novice_dataset.npz"
        dataset_path = resolve_dataset_path(default_path)

    dataset = VizDoomDataset(str(dataset_path), sequence_length=args.sequence_length)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = VizDoomConvLSTM(
        dropout=args.dropout,
        hidden_channels=args.hidden_channels,
        num_layers=args.convlstm_layers
    ).to(device)

    button_criterion = nn.BCEWithLogitsLoss()
    mouse_criterion = SignedMSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')

    print(f"Training {args.dataset_type} ConvLSTM model on {device}")
    print(f"Dataset: {dataset_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, button_criterion, mouse_criterion)
        val_loss = validate(model, val_loader, device, button_criterion, mouse_criterion)

        print(f"Epoch {epoch + 1}/{args.num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"{args.dataset_type}_convlstm_best.pth"))
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")

        torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f"{args.dataset_type}_convlstm_last.pth"))

    print("Training complete!")


if __name__ == "__main__":
    main()
