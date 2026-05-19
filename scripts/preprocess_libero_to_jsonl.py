"""
LIBERO HDF5 -> LAPA JSONL converter.

Walks a directory of LIBERO *.hdf5 files (libero_90 / libero_10 layout), and
emits a JSONL file matching the schema consumed by `latent_pretraining.train`
with `train_dataset.type='json_vision_delta_action'` and processor flags
`fields_from_example='fields'`, `n_tokens_per_action=7`, `n_tokens_per_frame=256`,
`max_n_frames=1`, `img_aug=False`.

Per-record schema (one record per (demo, timestep)):
  {
    "instruction": "<s> You are a helpful assistant. USER: What action should the robot take to `<task>` ASSISTANT:",
    "raw_actions": [7 floats],          # original LIBERO 7-dim action (xyz, rxyz, gripper)
    "vision":      ["<256 str ints>"],  # VQGAN tokens for agentview_rgb at this timestep
    "action":      ["<7 str ints>"],    # bin indices in [0, action_vocab_size-1]
    "fields":      "[instruction],[vision],action"
  }

Side outputs:
  bins.csv          -- per-dim discretization edges (compatible with data/simpler.csv)
  manifest.json     -- counts, splits, source file -> instruction map, action stats

Action discretization:
  - dims 0..5 (xyz + rxyz):  pd.qcut(action_vocab_size, duplicates='drop')
  - dim 6 (gripper):         LIBERO gripper in {-1, +1} -> {0, 1}; saved as bins [-1.5, 0, 1.5]

Memory:
  - Loads one HDF5 at a time, batches RGB through VQGAN in chunks of --vqgan-batch frames.
  - Pass 1 collects raw actions (in-memory ~50 MB max), then computes qcut bins once.
  - Pass 2 streams frames -> VQGAN -> JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple

import albumentations as A
import h5py
import numpy as np
import pandas as pd


# --- helpers ----------------------------------------------------------------


def filename_to_instruction(filename: str) -> str:
    """Turn 'KITCHEN_SCENE10_close_the_top_drawer_of_the_cabinet_demo.hdf5'
    into 'close the top drawer of the cabinet'.
    LIBERO file convention: <SCENE_GROUP>_SCENE<n>_<task>_demo.hdf5
    """
    stem = Path(filename).stem  # drop .hdf5
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    parts = stem.split("_")
    # drop leading SCENE_GROUP tokens (uppercase) and SCENE<NUM>
    i = 0
    while i < len(parts) and (parts[i].isupper() or parts[i].startswith("SCENE")):
        i += 1
    task_parts = parts[i:]
    return " ".join(task_parts).lower().replace("  ", " ").strip()


def assign_bin(x: float, bins: np.ndarray) -> int:
    """Assign x to a bin index in [0, len(bins) - 2]."""
    if x <= bins[0]:
        return 0
    if x >= bins[-1]:
        return len(bins) - 2
    # bins is sorted; np.searchsorted with 'right' gives index of upper edge
    idx = int(np.searchsorted(bins, x, side="right")) - 1
    return max(0, min(len(bins) - 2, idx))


def build_instruction(task: str) -> str:
    return (
        "<s> You are a helpful assistant. USER: What action should the robot "
        f"take to `{task}` ASSISTANT:"
    )


_VQGAN_PREPROCESSOR = A.Compose([
    A.LongestMaxSize(max_size=256),
    A.Resize(256, 256),
])


def preprocess_frames_for_vqgan(frames: np.ndarray) -> np.ndarray:
    """frames: [N, H, W, 3] uint8 -> [N, 256, 256, 3] float32 in [-1, 1].

    Bug A fix: rotate 180° (HDF5 GL convention -> EGL runtime convention),
               matching the flip in rollout_eval_libero10.py.
    Bug B fix: use Albumentations LongestMaxSize+Resize to exactly match the
               sampler's _process_frame() in sampler_latent_action_pretrain.py.
    """
    n = frames.shape[0]
    out = np.empty((n, 256, 256, 3), dtype=np.float32)
    for i in range(n):
        img = frames[i, ::-1, ::-1]  # Bug A: rotate 180° (flip both axes)
        img = _VQGAN_PREPROCESSOR(image=img)["image"]  # Bug B: Albumentations
        out[i] = img.astype(np.float32)
    out = out / 127.5 - 1.0
    return out


# --- main passes ------------------------------------------------------------


def list_hdf5(input_dir: Path) -> List[Path]:
    files = sorted(p for p in input_dir.iterdir() if p.suffix == ".hdf5")
    if not files:
        raise FileNotFoundError(f"No .hdf5 files under {input_dir}")
    return files


def gather_actions(files: List[Path], train_demos_per_task: int) -> Tuple[np.ndarray, dict]:
    """Pass 1: collect all training-set raw actions into a single array.

    Returns:
      actions: (M, 7) float32
      per_file_demo_count: dict[str, int]
    """
    chunks = []
    per_file_demo_count: dict = {}
    for fp in files:
        with h5py.File(fp, "r") as f:
            demo_keys = sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1]))
            per_file_demo_count[fp.name] = len(demo_keys)
            train_keys = demo_keys[:train_demos_per_task]
            for k in train_keys:
                a = f[f"data/{k}/actions"][:]
                chunks.append(a.astype(np.float32))
    actions = np.concatenate(chunks, axis=0)
    return actions, per_file_demo_count


def compute_bins(
    actions: np.ndarray, action_vocab_size: int
) -> List[np.ndarray]:
    """For dims 0..5 use pd.qcut to get bin edges; for dim 6 (gripper) use
    fixed binary bins compatible with LAPA's simpler.csv convention.
    """
    bins: List[np.ndarray] = []
    for d in range(6):
        col = pd.Series(actions[:, d])
        _, edges = pd.qcut(
            col,
            q=action_vocab_size,
            labels=False,
            retbins=True,
            duplicates="drop",
        )
        edges = np.asarray(edges, dtype=np.float64)
        bins.append(edges)
    # gripper bins: maps -1 -> 0, +1 -> 1
    bins.append(np.array([-1.5, 0.0, 1.5], dtype=np.float64))
    return bins


def save_bins_csv(bins: List[np.ndarray], path: Path) -> None:
    df = pd.DataFrame(bins)
    df.to_csv(path, index=False)


def load_bins_csv(path: Path) -> List[np.ndarray]:
    """Load per-dim bin edges from a bins.csv written by save_bins_csv."""
    df = pd.read_csv(path)
    bins: List[np.ndarray] = []
    for _, row in df.iterrows():
        # Drop NaN padding and convert to float64 array
        vals = row.dropna().values.astype(np.float64)
        bins.append(vals)
    return bins


def init_vqgan(vqgan_ckpt: str):
    # Defer imports so the script can run --help / --action-stats-only without JAX.
    from latent_pretraining.vqgan import VQGAN

    print(f"[vqgan] loading from {vqgan_ckpt} ...", flush=True)
    vq = VQGAN(vqgan_ckpt, replicate=False)
    # warm-up
    import jax  # noqa: F401
    dummy = np.zeros((1, 256, 256, 3), dtype=np.float32)
    _ = encode_with_vqgan(vq, dummy)
    print("[vqgan] ready.", flush=True)
    return vq


def encode_with_vqgan(vq, pixel_values: np.ndarray) -> np.ndarray:
    """Returns int array [N, 256] of token IDs."""
    import jax
    enc = jax.device_get(vq.encode(pixel_values))[1].astype(np.int32)
    # enc is [N, 16, 16] -> flatten to [N, 256]
    return enc.reshape(enc.shape[0], -1)


def emit_jsonl(
    files: List[Path],
    bins: List[np.ndarray],
    vq,
    train_demos_per_task: int,
    val_demos_per_task: int,
    out_train: Path,
    out_val: Path,
    vqgan_batch: int,
    progress_every: int = 5,
) -> dict:
    counts = {"train": 0, "val": 0}
    per_task: dict = {}
    train_f = open(out_train, "w")
    val_f = open(out_val, "w")
    try:
        for ti, fp in enumerate(files):
            task = filename_to_instruction(fp.name)
            instruction = build_instruction(task)
            with h5py.File(fp, "r") as f:
                demo_keys = sorted(
                    f["data"].keys(), key=lambda s: int(s.split("_")[1])
                )
                splits = [
                    ("train", demo_keys[:train_demos_per_task], train_f),
                    (
                        "val",
                        demo_keys[
                            train_demos_per_task : train_demos_per_task
                            + val_demos_per_task
                        ],
                        val_f,
                    ),
                ]
                for split_name, keys, out_fh in splits:
                    for k in keys:
                        d = f[f"data/{k}"]
                        frames = d["obs/agentview_rgb"][:]  # [T, 128, 128, 3]
                        actions = d["actions"][:].astype(np.float32)  # [T, 7]
                        T = frames.shape[0]
                        # batched VQGAN
                        vision_tokens_all = np.empty((T, 256), dtype=np.int32)
                        for s in range(0, T, vqgan_batch):
                            e = min(s + vqgan_batch, T)
                            pix = preprocess_frames_for_vqgan(frames[s:e])
                            vision_tokens_all[s:e] = encode_with_vqgan(vq, pix)
                        # per-timestep emission
                        for t in range(T):
                            a = actions[t]
                            action_tokens = [
                                str(assign_bin(float(a[d]), bins[d])) for d in range(7)
                            ]
                            vision = [str(int(x)) for x in vision_tokens_all[t]]
                            rec = {
                                "instruction": instruction,
                                "raw_actions": [float(x) for x in a],
                                "vision": vision,
                                "action": action_tokens,
                                "fields": "[instruction],[vision],action",
                            }
                            out_fh.write(json.dumps(rec) + "\n")
                            counts[split_name] += 1
                per_task[fp.name] = {
                    "task": task,
                    "n_demos": len(demo_keys),
                }
            if (ti + 1) % progress_every == 0 or ti == len(files) - 1:
                print(
                    f"[convert] {ti + 1}/{len(files)} files | "
                    f"train={counts['train']} val={counts['val']}",
                    flush=True,
                )
    finally:
        train_f.close()
        val_f.close()
    return {"counts": counts, "per_task": per_task}


# --- entrypoint -------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", type=Path, required=True,
                   help="Directory containing LIBERO *.hdf5 files.")
    p.add_argument("--bins-csv", type=Path, default=None,
                   help="If given, load bin edges from this CSV instead of computing from training demos. "
                        "Use the bins.csv from the training-set preprocessing run so that action "
                        "token indices are consistent with a checkpoint trained on those bins.")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Directory to write train.jsonl, val.jsonl, bins.csv, manifest.json.")
    p.add_argument("--vqgan-ckpt", type=str, required=True,
                   help="Path to LAPA vqgan checkpoint file.")
    p.add_argument("--action-vocab-size", type=int, default=245,
                   help="Number of bins for action discretization (must match llama.action_vocab_size).")
    p.add_argument("--train-demos-per-task", type=int, default=45)
    p.add_argument("--val-demos-per-task", type=int, default=5)
    p.add_argument("--vqgan-batch", type=int, default=64)
    p.add_argument("--limit-files", type=int, default=-1,
                   help="If > 0, only convert this many files (for smoke testing).")
    p.add_argument("--action-stats-only", action="store_true",
                   help="Only compute and save bins.csv + action stats; do not encode vision.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files = list_hdf5(args.input_dir)
    if args.limit_files > 0:
        files = files[: args.limit_files]
    print(f"[convert] {len(files)} hdf5 files in {args.input_dir}", flush=True)

    if args.bins_csv is not None:
        # Reuse bins from an existing preprocessing run (e.g., libero90) so that
        # action token indices are consistent with the checkpoint being evaluated.
        print(f"[pass 1] loading bins from {args.bins_csv} (skipping re-computation)", flush=True)
        bins = load_bins_csv(args.bins_csv)
        print(f"[pass 1] loaded {len(bins)} dim bins from CSV", flush=True)
        stats: dict = {"bin_counts": [int(len(b) - 1) for b in bins], "bins_source": str(args.bins_csv)}
    else:
        print("[pass 1] gathering training actions ...", flush=True)
        actions, per_file_demos = gather_actions(files, args.train_demos_per_task)
        print(
            f"[pass 1] {actions.shape[0]} train action vectors "
            f"({actions.shape[0] / max(1,len(files)):.0f} per task avg)",
            flush=True,
        )
        bins = compute_bins(actions, args.action_vocab_size)
        save_bins_csv(bins, args.output_dir / "bins.csv")
        stats = {
            "action_min": actions.min(axis=0).tolist(),
            "action_max": actions.max(axis=0).tolist(),
            "action_mean": actions.mean(axis=0).tolist(),
            "action_std": actions.std(axis=0).tolist(),
            "bin_counts": [int(len(b) - 1) for b in bins],
            "per_file_demos": per_file_demos,
            "args": {
                "action_vocab_size": args.action_vocab_size,
                "train_demos_per_task": args.train_demos_per_task,
                "val_demos_per_task": args.val_demos_per_task,
            },
        }
        print("[pass 1] action stats:")
        for k in ("action_min", "action_max", "action_mean", "action_std", "bin_counts"):
            print(f"   {k} = {stats[k]}")

    if args.action_stats_only:
        with open(args.output_dir / "manifest.json", "w") as fh:
            json.dump(stats, fh, indent=2)
        print("[done] action-stats-only complete.", flush=True)
        return 0

    print("[pass 2] initializing VQGAN ...", flush=True)
    vq = init_vqgan(args.vqgan_ckpt)

    out_train = args.output_dir / "train.jsonl"
    out_val = args.output_dir / "val.jsonl"
    emit_stats = emit_jsonl(
        files=files,
        bins=bins,
        vq=vq,
        train_demos_per_task=args.train_demos_per_task,
        val_demos_per_task=args.val_demos_per_task,
        out_train=out_train,
        out_val=out_val,
        vqgan_batch=args.vqgan_batch,
    )

    stats.update(emit_stats)
    with open(args.output_dir / "manifest.json", "w") as fh:
        json.dump(stats, fh, indent=2)

    print(
        f"[done] wrote {out_train} ({emit_stats['counts']['train']} lines) "
        f"and {out_val} ({emit_stats['counts']['val']} lines).",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
