from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch


def _as_coords(value) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 2 or array.shape[1] < 2:
        raise ValueError(f"Coordinates must have shape [N,2+], got {array.shape}")
    return np.ascontiguousarray(array[:, :2])


def load_coordinates(path: str | Path) -> np.ndarray:
    """Load level-0 coordinates from CLAM/Trident-style artifacts."""
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        import h5py

        with h5py.File(source, "r") as handle:
            if "coords" not in handle:
                raise KeyError(f"HDF5 file has no 'coords' dataset: {source}")
            return _as_coords(handle["coords"][:])
    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows and {"x", "y"}.issubset(rows[0]):
            return _as_coords([[row["x"], row["y"]] for row in rows])
        with source.open("r", encoding="utf-8", newline="") as handle:
            return _as_coords(list(csv.reader(handle)))
    if suffix == ".jsonl":
        with source.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        rows = [[row["x"], row["y"]] if isinstance(row, dict) else row for row in rows]
        return _as_coords(rows)
    if suffix == ".json":
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            payload = payload.get("coords", payload.get("coordinates"))
        return _as_coords(payload)
    if suffix in {".pt", ".pth"}:
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(source, map_location="cpu")
        if isinstance(payload, dict):
            payload = payload.get("coords", payload.get("coordinates"))
        if torch.is_tensor(payload):
            payload = payload.cpu().numpy()
        return _as_coords(payload)
    raise ValueError(f"Unsupported coordinate format: {source.suffix}")
