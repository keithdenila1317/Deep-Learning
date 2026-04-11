"""
Diagnostic inference script to debug model predictions.
Shows what actions the model predicts frame-by-frame.
"""
import torch
import numpy as np
from pathlib import Path
import sys
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from model import VizDoomCNN


def load_sequences_from_dataset(dataset_npz, sequence_length=4, num_samples=50):
    """Load sample sequences from dataset for testing."""
    data = np.load(dataset_npz)
    frames = data['frames']
    actions = data['actions']
    
    samples = []
    for i in range(0, len(frames) - sequence_length, max(1, len(frames) // (num_samples + 1))):
        if i + sequence_length < len(frames):
            frame_seq = frames[i:i+sequence_length].astype(np.float32) / 255.0
            frame_seq = np.expand_dims(frame_seq, axis=1)  # [T, 1, H, W]
            action_true = actions[i + sequence_length - 1]
            samples.append({
                'frames': torch.tensor(frame_seq, dtype=torch.float32),
                'buttons_true': torch.tensor(action_true[:5], dtype=torch.float32),
                'mouse_true': torch.tensor(action_true[5:] / 55.0, dtype=torch.float32),
            })
    return samples


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Load model
    model = VizDoomCNN().to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}\n")
    
    # Load test sequences
    print(f"Loading {args.num_samples} sample sequences from {args.dataset}...")
    samples = load_sequences_from_dataset(args.dataset, sequence_length=4, num_samples=args.num_samples)
    print(f"Loaded {len(samples)} sequences\n")
    
    # Analyze predictions
    button_names = ['ATTACK', 'MOVE_RIGHT', 'MOVE_LEFT', 'MOVE_BACKWARD', 'MOVE_FORWARD']
    button_accuracies = [0.0] * 5
    mouse_diffs = []
    
    print("=" * 80)
    print("INFERENCE DIAGNOSTICS")
    print("=" * 80)
    
    with torch.no_grad():
        for idx, sample in enumerate(samples[:min(10, len(samples))]):  # Show first 10
            frames = sample['frames'].unsqueeze(0).to(device)  # [1, T, 1, H, W]
            buttons_true = sample['buttons_true'].to(device)
            mouse_true = sample['mouse_true'].to(device)
            
            button_logits, mouse_pred = model(frames)
            button_probs = torch.sigmoid(button_logits[0])
            button_pred = (button_probs > 0.5).float()
            
            print(f"\nSample {idx + 1}:")
            print(f"  True buttons:  {[int(b) for b in buttons_true.cpu()]}")
            print(f"  Pred buttons:  {[int(b) for b in button_pred.cpu()]}")
            print(f"  Confidence:    {[f'{p:.2f}' for p in button_probs.cpu().tolist()]}")
            
            for b_idx in range(5):
                if buttons_true[b_idx] == button_pred[b_idx]:
                    button_accuracies[b_idx] += 1
            
            print(f"  True mouse:    [{mouse_true[0].item():+.3f}, {mouse_true[1].item():+.3f}]")
            print(f"  Pred mouse:    [{mouse_pred[0, 0].item():+.3f}, {mouse_pred[0, 1].item():+.3f}]")
            
            mouse_error = torch.abs(mouse_pred[0] - mouse_true).mean().item()
            mouse_diffs.append(mouse_error)
            print(f"  Mouse error:   {mouse_error:.5f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    num_test = min(10, len(samples))
    for b_idx in range(5):
        acc = button_accuracies[b_idx] / num_test * 100
        print(f"{button_names[b_idx]:15} accuracy: {acc:5.1f}%")
    
    if mouse_diffs:
        avg_mouse_error = np.mean(mouse_diffs)
        max_mouse_error = np.max(mouse_diffs)
        print(f"\nMouse MAE:      {avg_mouse_error:.5f}")
        print(f"Mouse max error: {max_mouse_error:.5f}")
    
    print("\n" + "=" * 80)
    print("DIAGNOSIS:")
    print("=" * 80)
    
    # Check if model is stuck predicting one action
    forward_preds = []
    with torch.no_grad():
        for sample in samples:
            frames = sample['frames'].unsqueeze(0).to(device)
            button_logits, _ = model(frames)
            button_pred = (torch.sigmoid(button_logits) > 0.5).float()[0]
            forward_preds.append(button_pred[4].item())  # MOVE_FORWARD
    
    forward_rate = sum(forward_preds) / len(forward_preds) * 100
    print(f"MOVE_FORWARD prediction rate: {forward_rate:.1f}%")
    
    if forward_rate > 70:
        print("⚠️  WARNING: Model predicts MOVE_FORWARD too often!")
        print("   This explains the 'going in circles' behavior.")
        print("   Root causes:")
        print("   1. Insufficient temporal sequences during training")
        print("   2. Training data heavily biased toward forward movement (39.29%!)")
        print("   3. Mouse control not being used effectively")
        print("\n✓ SOLUTIONS:")
        print("   - Retrain with --sequence_length=4 (using temporal context)")
        print("   - Collect more diverse expert data")
        print("   - Verify mouse predictions are non-zero and varied")
    else:
        print("✓ Model prediction distribution looks reasonable")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--dataset", type=str, required=True, help="Path to .npz dataset")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to analyze")
    
    args = parser.parse_args()
    main(args)
