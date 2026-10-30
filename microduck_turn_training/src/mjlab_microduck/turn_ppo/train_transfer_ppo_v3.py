"""Safely launch the isolated MicroDuck transfer PPO task."""

from __future__ import annotations

import os
import platform
import sys

from .register_transfer_v3_task import (
    TRANSFER_V3_TASK_ID,
    register_transfer_v3_task,
)


def main():
    help_requested = (
        "--help" in sys.argv or "-h" in sys.argv
    )

    if not help_requested:
        if "--allow-training" not in sys.argv:
            raise SystemExit(
                "[STOP] Training requires --allow-training"
            )

        if platform.system() == "Darwin":
            raise SystemExit(
                "[STOP] Training on macOS is disabled"
            )

        if not os.environ.get("CUDA_VISIBLE_DEVICES"):
            raise SystemExit(
                "[STOP] CUDA_VISIBLE_DEVICES must be configured"
            )

        import torch

        if not torch.cuda.is_available():
            raise SystemExit(
                "[STOP] PyTorch CUDA device unavailable"
            )

        if os.environ.get("MICRODUCK_TURN_PRETRAINED_ONNX"):
            raise SystemExit(
                "[STOP] Unset MICRODUCK_TURN_PRETRAINED_ONNX "
                "when resuming from a checkpoint"
            )

        sys.argv.remove("--allow-training")

    registered = register_transfer_v3_task()

    assert registered == TRANSFER_V3_TASK_ID, (
        "[STOP] Unexpected training task registered"
    )

    print(
        f"[TRANSFER_PPO] Registered: {registered}",
        flush=True,
    )

    from mjlab.scripts.train import main as mjlab_train_main

    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())
