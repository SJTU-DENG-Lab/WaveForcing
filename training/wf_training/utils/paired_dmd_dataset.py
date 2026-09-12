import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class PairedDMDLatentDataset(Dataset):
    """Load original-DMD ``(prompt, z_ref, y_ref)`` records."""

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"paired DMD manifest does not exist: {manifest_path}")

        with self.manifest_path.open(encoding="utf-8") as handle:
            self.records = [json.loads(line) for line in handle if line.strip()]
        if not self.records:
            raise ValueError(f"paired DMD manifest is empty: {manifest_path}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        item_path = Path(record["path"])
        if not item_path.is_absolute():
            item_path = self.manifest_path.parent / item_path
        item = torch.load(item_path, map_location="cpu", weights_only=True)

        prompt = item["prompt"]
        z_ref = item["z_ref"]
        y_ref = item["y_ref"]
        if tuple(z_ref.shape) != (21, 16, 60, 104):
            raise ValueError(f"bad z_ref shape in {item_path}: {tuple(z_ref.shape)}")
        if tuple(y_ref.shape) != (21, 16, 60, 104):
            raise ValueError(f"bad y_ref shape in {item_path}: {tuple(y_ref.shape)}")
        if not torch.isfinite(z_ref).all() or not torch.isfinite(y_ref).all():
            raise ValueError(f"non-finite paired tensor in {item_path}")
        if int(item["pair_index"]) != int(record["pair_index"]):
            raise ValueError(f"pair_index mismatch in {item_path}")
        if "seed" in record and int(item["seed"]) != int(record["seed"]):
            raise ValueError(f"pair seed mismatch in {item_path}")
        if "prompt_index" in record and item.get("prompt_index") != record["prompt_index"]:
            raise ValueError(f"prompt_index mismatch in {item_path}")

        return {
            "prompts": prompt,
            "z_ref": z_ref,
            "y_ref": y_ref,
            "pair_index": int(item["pair_index"]),
            "pair_seed": int(item["seed"]),
        }
