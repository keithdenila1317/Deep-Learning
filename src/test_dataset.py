from dataset import VizDoomDataset
from pathlib import Path


def resolve_dataset_path(file_name):
    """Resolve dataset path across common project layouts."""
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

novice_dataset_path = resolve_dataset_path("novice_dataset.npz")
expert_dataset_path = resolve_dataset_path("expert_dataset.npz")
novice_dataset = VizDoomDataset(novice_dataset_path)
expert_dataset = VizDoomDataset(expert_dataset_path)
print("Novice size:", len(novice_dataset))

print("Expert size:", len(expert_dataset))

novice_frame, novice_action = novice_dataset[0]
expert_frame, expert_action = expert_dataset[0]

print("Novice frame shape:", novice_frame.shape)
print("Novice action shape:", novice_action.shape)

print("Expert frame shape:", expert_frame.shape)
print("Expert action shape:", expert_action.shape)