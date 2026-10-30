"""Transfer walking ONNX weights into the turn PPO Actor.

This runner is opt-in. No official repository files are modified.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import numpy_helper

from mjlab_microduck.tasks import MicroduckOnPolicyRunner


TRANSFER_ONNX_ENV = "MICRODUCK_TURN_PRETRAINED_ONNX"


class MicroduckTurnTransferRunner(MicroduckOnPolicyRunner):
    """Initialize turn PPO Actor from an existing walking ONNX."""

    def __init__(
        self,
        env,
        train_cfg: dict,
        log_dir=None,
        device="cpu",
        **kwargs,
    ):
        super().__init__(env, train_cfg, log_dir, device, **kwargs)

        pretrained_path = os.environ.get(TRANSFER_ONNX_ENV)
        if pretrained_path:
            self.initialize_actor_from_onnx(Path(pretrained_path))

        self.freeze_actor_normalization()

    def initialize_actor_from_onnx(self, path: Path) -> None:
        """Copy ONNX weights into the existing PPO Actor parameters."""
        if not path.is_file():
            raise FileNotFoundError(f"Walking ONNX not found: {path}")

        model = onnx.load(str(path))
        onnx.checker.check_model(model)

        weights = {
            item.name: numpy_helper.to_array(item).copy()
            for item in model.graph.initializer
        }

        actor = self.alg.actor
        state = actor.state_dict()

        source_names = {
            "obs_normalizer._mean": "obs_normalizer._mean",
        }

        for index in (0, 2, 4, 6):
            for suffix in ("weight", "bias"):
                key = f"mlp.{index}.{suffix}"
                source_names[key] = key

        denominator = torch.from_numpy(weights["onnx::Div_24"])
        std = denominator - actor.obs_normalizer.eps

        if not torch.all(std > 0):
            raise ValueError("Invalid reconstructed observation std")

        new_state = dict(state)

        for target_key, source_key in source_names.items():
            value = torch.from_numpy(weights[source_key])
            if tuple(value.shape) != tuple(state[target_key].shape):
                raise ValueError(f"Shape mismatch: {target_key}")
            new_state[target_key] = value.to(
                device=state[target_key].device,
                dtype=state[target_key].dtype,
            )

        new_state["obs_normalizer._std"] = std.to(
            device=state["obs_normalizer._std"].device,
            dtype=state["obs_normalizer._std"].dtype,
        )
        new_state["obs_normalizer._var"] = new_state[
            "obs_normalizer._std"
        ].square()

        actor.load_state_dict(new_state, strict=True)

        # Verify deterministic inference on several synthetic inputs.
        session = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name

        rng = np.random.default_rng(42)
        test_inputs = [
            np.zeros((1, 61), dtype=np.float32),
            rng.normal(0, 0.1, (1, 61)).astype(np.float32),
            rng.normal(0, 1.0, (1, 61)).astype(np.float32),
        ]

        actor_device = next(actor.parameters()).device
        actor.eval()

        for i, x in enumerate(test_inputs):
            expected = session.run(None, {input_name: x})[0]

            with torch.no_grad():
                actual = actor.as_onnx(verbose=False)(
                    torch.from_numpy(x).to(actor_device)
                ).cpu().numpy()

            diff = np.abs(expected - actual).max()

            print(
                f"[turn-transfer] Sample {i}: "
                f"max_abs_diff={diff:.8f}"
            )

            if not np.allclose(
                expected, actual, rtol=1e-4, atol=1e-4
            ):
                raise RuntimeError(
                    f"ONNX-to-Actor inference mismatch: sample {i}"
                )

        print(
            "[turn-transfer] Walking ONNX Actor transfer verified: "
            f"{path}"
        )

    def freeze_actor_normalization(self) -> None:
        """Disable Actor statistics updates without freezing MLP weights."""
        normalizer = self.alg.actor.obs_normalizer

        def frozen_update(_observations):
            return None

        normalizer.update = frozen_update
        normalizer.eval()
        print("[turn-transfer] Actor normalization updates disabled")

    def load(self, path, *args, **kwargs):
        """Restore checkpoint, then reapply Actor normalization freeze."""
        if os.environ.get(TRANSFER_ONNX_ENV):
            raise RuntimeError(
                "Checkpoint resume and pretrained ONNX initialization "
                "must not be enabled simultaneously. Unset "
                f"{TRANSFER_ONNX_ENV} before loading a checkpoint."
            )

        infos = super().load(path, *args, **kwargs)

        # Checkpoint optimizer state may contain the original PPO learning
        # rate (0.001). Restore the requested transfer learning rate on
        # both PPO and optimizer before the first training update.
        requested_lr = float(self.cfg["algorithm"]["learning_rate"])
        if not (0.0 < requested_lr <= 1e-4):
            raise RuntimeError(
                f"Unsafe transfer PPO learning rate: {requested_lr}"
            )

        self.alg.learning_rate = requested_lr
        for group in self.alg.optimizer.param_groups:
            group["lr"] = requested_lr

        print(
            "[turn-transfer] Checkpoint resume learning rate restored: "
            f"{requested_lr:.8f}"
        )

        self.freeze_actor_normalization()
        print("[turn-transfer] Actor normalization frozen after checkpoint load")
        return infos
