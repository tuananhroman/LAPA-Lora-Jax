#!/usr/bin/env python3
"""
launch.py — translate a YAML config + env vars into the long CLI invocation
expected by latent_pretraining.train.

Usage:
    python scripts/launch.py configs/libero90_lora_v2.yaml [--dry-run] [--override key=value ...]

Environment variables consumed:
    WANDB_API_KEY, WANDB_PROJECT, WANDB_ENTITY, WANDB_MODE, WANDB_RUN_NAME
    CUDA_VISIBLE_DEVICES (passed through by docker / shell)
    LAPA_PYTHON           (path to python; defaults to current interpreter)
    LAPA_EXTRA_ARGS       (raw string appended to the CLI for debugging)
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


def _flatten_llama(cfg: dict) -> str:
    """Render the --update_llama_config=dict(...) argument."""
    items = []
    for k, v in cfg.items():
        if isinstance(v, str):
            items.append(f"{k}='{v}'")
        elif isinstance(v, bool):
            items.append(f"{k}={'True' if v else 'False'}")
        else:
            items.append(f"{k}={v}")
    return "dict(" + ", ".join(items) + ")"


def _apply_overrides(cfg: dict, overrides: list[str]) -> None:
    """Apply `--override a.b.c=value` style edits in-place."""
    for ov in overrides:
        if "=" not in ov:
            raise SystemExit(f"--override expects key=value, got {ov!r}")
        key, raw = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        # naive type coercion
        try:
            val: object = yaml.safe_load(raw)
        except yaml.YAMLError:
            val = raw
        node[parts[-1]] = val


def build_args(cfg: dict) -> list[str]:
    model, ckpt, data, train, opt, lora, llama, log = (
        cfg["model"], cfg["checkpoint"], cfg["data"],
        cfg["train"], cfg["optimizer"], cfg["lora"],
        cfg["llama"], cfg["logger"],
    )

    # Honour WANDB_MODE: offline/disabled → logger.online=False
    wandb_mode = os.environ.get("WANDB_MODE", "online").lower()
    online = bool(log.get("online", True)) and wandb_mode == "online"

    # Allow env-var overrides for the most useful logger fields
    project = os.environ.get("WANDB_PROJECT", log["project_id"])
    run_name = os.environ.get("WANDB_RUN_NAME", log["experiment_id"])

    args = [
        f"--modality={train['modality']}",
        f"--mesh_dim={train['mesh_dim']}",
        f"--dtype={train['dtype']}",
        f"--total_steps={train['total_steps']}",
        f"--log_freq={train['log_freq']}",
        f"--eval_steps={train['eval_steps']}",
        f"--eval_log_freq={train['eval_log_freq']}",
        f"--save_model_freq={train['save_model_freq']}",
        f"--save_milestone_freq={ckpt['save_milestone_freq']}",
        f"--load_llama_config={model['config']}",
        f"--load_checkpoint={ckpt['load']}",
        f"--update_llama_config={_flatten_llama({**llama, 'action_vocab_size': model['action_vocab_size'], 'delta_vocab_size': model['delta_vocab_size']})}",
        f"--tokenizer.vocab_file={ckpt['tokenizer']}",

        f"--optimizer.type={opt['type']}",
        f"--optimizer.adamw_optimizer.bf16_momentum={opt['bf16_momentum']}",
        f"--optimizer.adamw_optimizer.lr={opt['lr']}",
        f"--optimizer.adamw_optimizer.end_lr={opt['end_lr']}",
        f"--optimizer.adamw_optimizer.lr_warmup_steps={opt['lr_warmup_steps']}",
        f"--optimizer.adamw_optimizer.lr_decay_steps={opt['lr_decay_steps']}",
        f"--optimizer.adamw_optimizer.weight_decay={opt['weight_decay']}",
        f"--optimizer.adamw_optimizer.clip_gradient={opt['clip_gradient']}",
        f"--optimizer.accumulate_gradient_steps={opt['accumulate_gradient_steps']}",

        f"--lora_only={1 if lora['enabled'] else 0}",
        f"--llama.lora_rank={lora['rank']}",
        f"--llama.lora_alpha={lora['alpha']}",
        f"--llama.use_rslora={lora['use_rslora']}",
        f"--lora_plus_ratio={lora['plus_ratio']}",
        f"--lora_train_patterns={lora['train_patterns']}",
        f"--heads_lr_multiplier={lora['heads_lr_multiplier']}",
        f"--llama.action_vocab_size={model['action_vocab_size']}",
        f"--llama.delta_vocab_size={model['delta_vocab_size']}",

        f"--train_dataset.type={data['type']}",
        "--train_dataset.delta_vision_action_processor.fields_from_example=fields",
        f"--train_dataset.delta_vision_action_processor.n_tokens_per_action={data['n_tokens_per_action']}",
        f"--train_dataset.delta_vision_action_processor.n_tokens_per_delta={data['n_tokens_per_delta']}",
        f"--train_dataset.delta_vision_action_processor.img_aug={data['img_aug']}",
        f"--train_dataset.delta_vision_action_processor.max_n_frames={data['max_n_frames']}",
        "--train_dataset.json_delta_action_dataset.mode=pad",
        f"--train_dataset.json_delta_action_dataset.path={data['path']}",
        f"--train_dataset.json_delta_action_dataset.seq_length={data['seq_length']}",
        f"--train_dataset.json_delta_action_dataset.batch_size={data['batch_size']}",
        "--train_dataset.json_delta_action_dataset.tokenizer_processes=4",
        "--train_dataset.json_delta_action_dataset.tokenizer_parallel_chunk_size=32",
        "--train_dataset.json_delta_action_dataset.tokenizer_parallel_batch_size=32",
        f"--train_dataset.json_delta_action_dataset.use_data_sharded_loader={data['use_data_sharded_loader']}",
        f"--use_data_sharded_loader={data['use_data_sharded_loader']}",

        f"--checkpointer.save_optimizer_state={ckpt['save_optimizer_state']}",
        f"--autoresume={ckpt['autoresume']}",

        f"--logger.online={online}",
        f"--logger.append_uuid={log['append_uuid']}",
        f"--logger.project_id={project}",
        f"--logger.experiment_id={run_name}",
        f"--logger.experiment_note={log['experiment_note']}",
        f"--logger.output_dir={log['output_dir']}",
        f"--logger.wandb_dir={log['wandb_dir']}",
    ]
    return args


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    cfg_path = Path(sys.argv[1])
    rest = sys.argv[2:]
    dry_run = "--dry-run" in rest
    overrides = [rest[i + 1] for i, a in enumerate(rest) if a == "--override" and i + 1 < len(rest)]

    cfg = yaml.safe_load(cfg_path.read_text())
    if overrides:
        _apply_overrides(cfg, overrides)

    python = os.environ.get("LAPA_PYTHON", sys.executable)
    cmd = [python, "-u", "-m", "latent_pretraining.train", *build_args(cfg)]

    extra = os.environ.get("LAPA_EXTRA_ARGS", "").strip()
    if extra:
        cmd.extend(shlex.split(extra))

    print("=" * 78, flush=True)
    print(f"  config : {cfg_path}", flush=True)
    print(f"  python : {python}", flush=True)
    print(f"  wandb  : mode={os.environ.get('WANDB_MODE', 'online')}  "
          f"project={os.environ.get('WANDB_PROJECT', cfg['logger']['project_id'])}  "
          f"run={os.environ.get('WANDB_RUN_NAME', cfg['logger']['experiment_id'])}", flush=True)
    print(f"  GPUs   : {os.environ.get('CUDA_VISIBLE_DEVICES', '(all visible)')}", flush=True)
    print(f"  resume : {cfg['checkpoint']['load']}", flush=True)
    print("=" * 78, flush=True)
    print("CMD: " + " ".join(shlex.quote(c) for c in cmd), flush=True)

    if dry_run:
        return 0
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
