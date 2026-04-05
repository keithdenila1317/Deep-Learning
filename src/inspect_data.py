import numpy as np
from pathlib import Path


def resolve_dataset_path(file_name):
    """Resolve dataset path across project"""
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

    checked_paths = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not find '{file_name}'. Checked:\n{checked_paths}"
    )


def is_binary_column(col):
    unique_vals = np.unique(col)
    return np.all(np.isin(unique_vals, [0, 1]))


def inspect_action_columns(actions, dataset_name):
    print("\n" + "=" * 60)
    print(f"{dataset_name} ACTION COLUMN INSPECTION")
    print("=" * 60)

    num_cols = actions.shape[1]

    binary_cols = []
    continuous_cols = []

    for i in range(num_cols):
        col = actions[:, i]
        unique_vals = np.unique(col)

        print("\n" + "-" * 40)
        print(f"Column {i}")
        print(f"Min: {col.min()}")
        print(f"Max: {col.max()}")
        print(f"Mean: {col.mean():.4f}")
        print(f"Std: {col.std():.4f}")
        print(f"Unique values sample (first 20): {unique_vals[:20]}")

        if is_binary_column(col):
            binary_cols.append(i)
            ones = np.sum(col == 1)
            zeros = np.sum(col == 0)
            print("Type guess: BINARY")
            print(f"Zeros: {zeros}")
            print(f"Ones: {ones}")
            print(f"Percent ones: {ones / len(col):.4%}")
        else:
            continuous_cols.append(i)
            print("Type guess: CONTINUOUS")

    print("\n" + "-" * 40)
    print(f"Binary columns guessed: {binary_cols}")
    print(f"Continuous columns guessed: {continuous_cols}")

    if continuous_cols:
        print("\nContinuous column stats:")
        for i in continuous_cols:
            col = actions[:, i]
            print(
                f"Column {i}: min={col.min():.4f}, max={col.max():.4f}, "
                f"mean={col.mean():.4f}, std={col.std():.4f}"
            )


def inspect_sample_pairs(frames, actions, dataset_name, num_samples=5):
    """Print a few frame-action pairs."""
    print("\n" + "=" * 60)
    print(f"{dataset_name} SAMPLE FRAME-ACTION PAIRS")
    print("=" * 60)

    max_samples = min(num_samples, len(frames))

    for i in range(max_samples):
        frame = frames[i]
        action = actions[i]

        print("\n" + "-" * 40)
        print(f"Sample index: {i}")
        print(f"Frame shape: {frame.shape}")
        print(f"Frame dtype: {frame.dtype}")
        print(f"Frame min: {frame.min()}, max: {frame.max()}, mean: {frame.mean():.4f}")
        print(f"Action: {action}")


def inspect_npz(npz_path, name):
    """Inspect one .npz dataset"""
    print("=" * 60)
    print(f"Inspecting {name}: {npz_path}")
    print("=" * 60)

    data = np.load(npz_path, allow_pickle=True)

    print("\nTop-level keys:")
    print(data.files)

    # Gen inspection
    for key in data.files:
        arr = data[key]

        print("\n" + "-" * 40)
        print(f"Key: {key}")
        print(f"Type: {type(arr)}")
        print(f"Shape: {getattr(arr, 'shape', 'No shape')}")
        print(f"Dtype: {getattr(arr, 'dtype', 'No dtype')}")

        if isinstance(arr, np.ndarray) and arr.size > 0:
            first_item = arr[0]

            print(f"First item type: {type(first_item)}")

            if hasattr(first_item, "shape"):
                print(f"First item shape: {first_item.shape}")

            if np.issubdtype(arr.dtype, np.number):
                try:
                    print(f"Min value: {np.min(arr)}")
                    print(f"Max value: {np.max(arr)}")
                    print(f"Mean value: {np.mean(arr):.4f}")
                except Exception as e:
                    print(f"Could not compute stats: {e}")

            try:
                preview = first_item
                if isinstance(preview, np.ndarray):
                    flat = preview.flatten()
                    print(f"First item preview (first 10 values): {flat[:10]}")
                else:
                    print(f"First item preview: {preview}")
            except Exception as e:
                print(f"Could not preview first item: {e}")

    
    if "frames" in data.files and "actions" in data.files:
        frames = data["frames"]
        actions = data["actions"]

        print("\n" + "=" * 60)
        print(f"{name} FRAME/ACTION ALIGNMENT CHECK")
        print("=" * 60)
        print(f"Frames shape: {frames.shape}")
        print(f"Actions shape: {actions.shape}")
        print(f"Same number of samples: {len(frames) == len(actions)}")

        inspect_action_columns(actions, name)
        inspect_sample_pairs(frames, actions, name, num_samples=5)

    print("\n")


if __name__ == "__main__":
    novice_path = resolve_dataset_path("novice_dataset.npz")
    expert_path = resolve_dataset_path("expert_dataset.npz")

    inspect_npz(novice_path, "NOVICE")
    inspect_npz(expert_path, "EXPERT")