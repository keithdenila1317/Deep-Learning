
import argparse

import torch
from torch.utils.data import DataLoader

from data_loader import Augmentation
from model import VizDoomCNN

def evaluate_model(model, loader, device):
    model.eval()

    total_samples = 0
    total_button_correct = 0
    total_button_labels = 0
    total_exact_match = 0
    total_mouse_mse = 0.0
    total_mouse_values = 0
    first_batch = None

    with torch.no_grad():
        for frames, button_true, mouse_true in loader:
            frames = frames.to(device)
            button_true = button_true.to(device)
            mouse_true = mouse_true.to(device)

            button_logits, mouse_output = model(frames)
            button_probs = torch.sigmoid(button_logits)
            button_pred = (button_probs > 0.5).float()

            if first_batch is None:
                first_batch = {
                    "button_probs": button_probs[:8].cpu(),
                    "mouse_output": mouse_output[:8].cpu(),
                    "actions": torch.cat([button_true[:8], mouse_true[:8]], dim=1).cpu(),
                }

            total_samples += frames.size(0)
            total_button_correct += (button_pred == button_true).sum().item()
            total_button_labels += button_true.numel()
            total_exact_match += (button_pred == button_true).all(dim=1).sum().item()
            total_mouse_mse += torch.sum((mouse_output - mouse_true) ** 2).item()
            total_mouse_values += mouse_true.numel()

    metrics = {
        "button_accuracy": total_button_correct / total_button_labels if total_button_labels else 0.0,
        "button_exact_match": total_exact_match / total_samples if total_samples else 0.0,
        "mouse_mse": total_mouse_mse / total_mouse_values if total_mouse_values else 0.0,
        "samples": total_samples,
        "first_batch": first_batch,
    }

    return metrics


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    novice_dataset = Augmentation(args.novice_dataset_path, is_train=False)
    expert_dataset = Augmentation(args.expert_dataset_path, is_train=False)

    novice_loader = DataLoader(novice_dataset, batch_size=args.batch_size, shuffle=False)
    expert_loader = DataLoader(expert_dataset, batch_size=args.batch_size, shuffle=False)

    novice_model = VizDoomCNN().to(device)
    novice_model.load_state_dict(torch.load(args.novice_checkpoint, map_location=device))

    expert_model = VizDoomCNN().to(device)
    expert_model.load_state_dict(torch.load(args.expert_checkpoint, map_location=device))

    novice_metrics = evaluate_model(novice_model, novice_loader, device)
    expert_metrics = evaluate_model(expert_model, expert_loader, device)

    print("=" * 60)
    print("NOVICE MODEL")
    print("=" * 60)
    print(f"Samples evaluated: {novice_metrics['samples']}")
    print(f"Button accuracy:   {novice_metrics['button_accuracy']:.2%}")
    print(f"Exact match rate:  {novice_metrics['button_exact_match']:.2%}")
    print(f"Mouse MSE:         {novice_metrics['mouse_mse']:.6f}")

    print("\nSample predictions:")
    print("Predicted button probabilities:")
    print(novice_metrics["first_batch"]["button_probs"])
    print("\nPredicted mouse output:")
    print(novice_metrics["first_batch"]["mouse_output"])
    print("\nTrue actions:")
    print(novice_metrics["first_batch"]["actions"])

    print("\n" + "=" * 60)
    print("EXPERT MODEL")
    print("=" * 60)
    print(f"Samples evaluated: {expert_metrics['samples']}")
    print(f"Button accuracy:   {expert_metrics['button_accuracy']:.2%}")
    print(f"Exact match rate:  {expert_metrics['button_exact_match']:.2%}")
    print(f"Mouse MSE:         {expert_metrics['mouse_mse']:.6f}")

    print("\nSample predictions:")
    print("Predicted button probabilities:")
    print(expert_metrics["first_batch"]["button_probs"])
    print("\nPredicted mouse output:")
    print(expert_metrics["first_batch"]["mouse_output"])
    print("\nTrue actions:")
    print(expert_metrics["first_batch"]["actions"])

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Novice Button Accuracy:   {novice_metrics['button_accuracy']:.2%}")
    print(f"Expert Button Accuracy:   {expert_metrics['button_accuracy']:.2%}")
    print(f"Novice Exact Match Rate:   {novice_metrics['button_exact_match']:.2%}")
    print(f"Expert Exact Match Rate:   {expert_metrics['button_exact_match']:.2%}")
    print(f"Novice Mouse MSE:          {novice_metrics['mouse_mse']:.6f}")
    print(f"Expert Mouse MSE:          {expert_metrics['mouse_mse']:.6f}")

    # Save results to file
    results_file = args.output_file
    with open(results_file, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("NOVICE MODEL\n")
        f.write("=" * 60 + "\n")
        f.write(f"Button accuracy:   {novice_metrics['button_accuracy']:.2%}\n")
        f.write(f"Exact match rate:  {novice_metrics['button_exact_match']:.2%}\n")
        f.write(f"Mouse MSE:         {novice_metrics['mouse_mse']:.6f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("EXPERT MODEL\n")
        f.write("=" * 60 + "\n")
        f.write(f"Button accuracy:   {expert_metrics['button_accuracy']:.2%}\n")
        f.write(f"Exact match rate:  {expert_metrics['button_exact_match']:.2%}\n")
        f.write(f"Mouse MSE:         {expert_metrics['mouse_mse']:.6f}\n")
        f.write("\n" + "=" * 60 + "\n")
        f.write("COMPARISON\n")
        f.write("=" * 60 + "\n")
        f.write(f"Novice Button Accuracy:   {novice_metrics['button_accuracy']:.2%}\n")
        f.write(f"Expert Button Accuracy:   {expert_metrics['button_accuracy']:.2%}\n")
        f.write(f"Novice Exact Match Rate:   {novice_metrics['button_exact_match']:.2%}\n")
        f.write(f"Expert Exact Match Rate:   {expert_metrics['button_exact_match']:.2%}\n")
        f.write(f"Novice Mouse MSE:          {novice_metrics['mouse_mse']:.6f}\n")
        f.write(f"Expert Mouse MSE:          {expert_metrics['mouse_mse']:.6f}\n")

    print(f"\nResults saved to {results_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--novice_dataset_path", type=str, default="data/raw/novice_dataset.npz")
    parser.add_argument("--expert_dataset_path", type=str, default="data/raw/expert_dataset.npz")
    parser.add_argument("--novice_checkpoint", type=str, default="checkpoints/novice_model_best.pth")
    parser.add_argument("--expert_checkpoint", type=str, default="checkpoints/expert_model_best.pth")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_file", type=str, default="evaluation_results.txt")
    args = parser.parse_args()
    main(args)