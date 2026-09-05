"""The sequential MLP intrusion detector (Module 3, block 2).

Architecture, as specified by the project:

    Input(n_features)
      -> Dense(64)  relu      (input projection)
      -> Dense(32)  tanh
      -> Dense(16)  relu
      -> Dense(8)   tanh
      -> Dense(1)   sigmoid   (benign / malicious probability)

Two interchangeable backends implement it:

``numpy``
    A self-contained implementation (forward/backward pass, Adam, mini-batch
    binary cross-entropy).  No deep-learning framework required, fully
    deterministic under a seed, and fast enough for the whole experiment grid.
``keras``
    The same topology as a ``tf.keras.Sequential`` model, for parity with the
    original paper's toolchain.

Select one with ``ModelConfig.backend``.  ``auto`` resolves to ``numpy``,
which is always available and much faster at this model size; set
``backend: keras`` to cross-check the same experiment on TensorFlow.
"""

from __future__ import annotations

import importlib.util
from typing import Protocol, runtime_checkable

import numpy as np

from ..config import ModelConfig


@runtime_checkable
class Classifier(Protocol):
    """The minimal surface the evaluation engine and defence rely on."""

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = ...,
        y_val: np.ndarray | None = ...,
        sample_weight: np.ndarray | None = ...,
    ) -> "Classifier": ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...

    def predict(self, X: np.ndarray, threshold: float = ...) -> np.ndarray: ...


# --------------------------------------------------------------------- numpy


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0.0).astype(z.dtype)


def _tanh_grad(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {
    "relu": (_relu, _relu_grad),
    "tanh": (np.tanh, _tanh_grad),
    "linear": (lambda z: z, lambda z: np.ones_like(z)),
}


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Branch-free stable sigmoid.
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    exp_z = np.exp(z[~pos])
    out[~pos] = exp_z / (1.0 + exp_z)
    return out


class NumpyMLP:
    """Dependency-light implementation of the sequential MLP."""

    backend = "numpy"

    def __init__(self, n_features: int, cfg: ModelConfig, seed: int = 0) -> None:
        if cfg.output_activation != "sigmoid":
            raise ValueError("the NumPy backend implements a sigmoid output only")
        self.cfg = cfg
        self.n_features = int(n_features)
        self.rng = np.random.default_rng(seed)
        self.units = tuple(cfg.hidden_units)
        self.acts = tuple(cfg.activations)
        for name in self.acts:
            if name not in ACTIVATIONS:
                raise ValueError(f"unsupported activation '{name}'")
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        self.history: dict[str, list[float]] = {"loss": [], "val_loss": [], "val_accuracy": []}
        self._build()

    def _build(self) -> None:
        sizes = (self.n_features, *self.units, 1)
        for i, (fan_in, fan_out) in enumerate(zip(sizes[:-1], sizes[1:])):
            act = self.acts[i] if i < len(self.acts) else "sigmoid"
            if act == "relu":  # He normal
                scale = np.sqrt(2.0 / fan_in)
                W = self.rng.normal(0.0, scale, (fan_in, fan_out))
            else:  # Glorot uniform, as Keras uses for tanh/sigmoid Dense layers
                limit = np.sqrt(6.0 / (fan_in + fan_out))
                W = self.rng.uniform(-limit, limit, (fan_in, fan_out))
            self.W.append(W)
            self.b.append(np.zeros((1, fan_out)))
        self._reset_optimiser()

    def _reset_optimiser(self) -> None:
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._t = 0

    # ------------------------------------------------------------- forward

    def _forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Return the pre-activations and activations of every layer."""
        pre: list[np.ndarray] = []
        acts: list[np.ndarray] = [X]
        h = X
        n_hidden = len(self.W) - 1
        for i in range(n_hidden):
            z = h @ self.W[i] + self.b[i]
            pre.append(z)
            h = ACTIVATIONS[self.acts[i]][0](z)
            acts.append(h)
        z = h @ self.W[-1] + self.b[-1]
        pre.append(z)
        acts.append(_sigmoid(z))
        return pre, acts

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, acts = self._forward(np.asarray(X, dtype=float))
        return acts[-1].ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    # -------------------------------------------------------------- training

    @staticmethod
    def _bce(y_true: np.ndarray, p: np.ndarray, w: np.ndarray | None = None) -> float:
        p = np.clip(p, 1e-9, 1.0 - 1e-9)
        terms = -(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
        if w is None:
            return float(terms.mean())
        return float(np.sum(terms * w) / max(np.sum(w), 1e-12))

    def _adam_step(self, grads_W: list[np.ndarray], grads_b: list[np.ndarray]) -> None:
        lr, b1, b2, eps = self.cfg.learning_rate, 0.9, 0.999, 1e-7
        self._t += 1
        bias1 = 1.0 - b1**self._t
        bias2 = 1.0 - b2**self._t
        for i in range(len(self.W)):
            self._mW[i] = b1 * self._mW[i] + (1 - b1) * grads_W[i]
            self._vW[i] = b2 * self._vW[i] + (1 - b2) * grads_W[i] ** 2
            self.W[i] -= lr * (self._mW[i] / bias1) / (np.sqrt(self._vW[i] / bias2) + eps)
            self._mb[i] = b1 * self._mb[i] + (1 - b1) * grads_b[i]
            self._vb[i] = b2 * self._vb[i] + (1 - b2) * grads_b[i] ** 2
            self.b[i] -= lr * (self._mb[i] / bias1) / (np.sqrt(self._vb[i] / bias2) + eps)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "NumpyMLP":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)
        w = (
            np.asarray(sample_weight, dtype=float).reshape(-1, 1)
            if sample_weight is not None
            else None
        )
        n = X.shape[0]
        batch = min(self.cfg.batch_size, n)
        patience = self.cfg.early_stopping_patience
        best_loss, best_state, waited = np.inf, None, 0

        for epoch in range(self.cfg.epochs):
            order = self.rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch):
                idx = order[start : start + batch]
                xb, yb = X[idx], y[idx]
                wb = w[idx] if w is not None else None
                pre, acts = self._forward(xb)
                p = acts[-1]
                m = xb.shape[0]
                # dL/dz for sigmoid + binary cross-entropy collapses to (p - y).
                delta = p - yb
                if wb is not None:
                    delta = delta * wb / max(wb.mean(), 1e-12)
                delta = delta / m
                grads_W: list[np.ndarray] = [None] * len(self.W)  # type: ignore[list-item]
                grads_b: list[np.ndarray] = [None] * len(self.b)  # type: ignore[list-item]
                for i in range(len(self.W) - 1, -1, -1):
                    grads_W[i] = acts[i].T @ delta + self.cfg.l2 * self.W[i]
                    grads_b[i] = delta.sum(axis=0, keepdims=True)
                    if i > 0:
                        delta = (delta @ self.W[i].T) * ACTIVATIONS[self.acts[i - 1]][1](pre[i - 1])
                self._adam_step(grads_W, grads_b)
                epoch_loss += self._bce(yb, p, wb) * m

            self.history["loss"].append(epoch_loss / n)
            if X_val is not None and y_val is not None:
                p_val = self.predict_proba(X_val)
                val_loss = self._bce(np.asarray(y_val, dtype=float), p_val)
                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(
                    float(np.mean((p_val >= 0.5).astype(int) == np.asarray(y_val)))
                )
                if patience > 0:
                    if val_loss < best_loss - 1e-6:
                        best_loss, waited = val_loss, 0
                        best_state = ([w_.copy() for w_ in self.W], [b_.copy() for b_ in self.b])
                    else:
                        waited += 1
                        if waited >= patience:
                            break
            if self.cfg.verbose and epoch % max(self.cfg.epochs // 10, 1) == 0:
                print(f"epoch {epoch:4d} loss={self.history['loss'][-1]:.4f}")

        if patience > 0 and best_state is not None:
            self.W, self.b = best_state
        return self

    def n_parameters(self) -> int:
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))


# --------------------------------------------------------------------- keras


class KerasMLP:
    """The same topology built with ``tf.keras.Sequential``."""

    backend = "keras"

    def __init__(self, n_features: int, cfg: ModelConfig, seed: int = 0) -> None:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
        self.tf = tf
        self.cfg = cfg
        self.n_features = int(n_features)
        layers = [tf.keras.layers.Input(shape=(n_features,))]
        for units, act in zip(cfg.hidden_units, cfg.activations):
            layers.append(tf.keras.layers.Dense(int(units), activation=act))
        layers.append(tf.keras.layers.Dense(1, activation=cfg.output_activation))
        self.model = tf.keras.Sequential(layers)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        self.history: dict[str, list[float]] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "KerasMLP":
        callbacks = []
        if self.cfg.early_stopping_patience > 0 and X_val is not None:
            callbacks.append(
                self.tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=self.cfg.early_stopping_patience,
                    restore_best_weights=True,
                )
            )
        hist = self.model.fit(
            np.asarray(X, dtype=float),
            np.asarray(y, dtype=float),
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=self.cfg.epochs,
            batch_size=self.cfg.batch_size,
            sample_weight=sample_weight,
            verbose=self.cfg.verbose,
            callbacks=callbacks,
        )
        self.history = {k: [float(x) for x in v] for k, v in hist.history.items()}
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(np.asarray(X, dtype=float), verbose=0).ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)

    def n_parameters(self) -> int:
        return int(self.model.count_params())


def keras_available() -> bool:
    """True when TensorFlow can be imported in this environment."""
    return importlib.util.find_spec("tensorflow") is not None


def resolve_backend(name: str) -> str:
    """Turn ``auto`` into a concrete backend name.

    ``auto`` picks NumPy: it is always available, bit-for-bit reproducible
    under a seed, and roughly eight times faster than Keras on a model this
    small (Keras' per-step graph overhead dominates a 3k-parameter network).
    Ask for ``keras`` explicitly to cross-check against the TensorFlow stack.
    """
    if name == "auto":
        return "numpy"
    if name == "keras" and not keras_available():
        raise RuntimeError(
            "model.backend='keras' but TensorFlow is not installed; "
            "install tensorflow-cpu or use backend='numpy'"
        )
    return name


def build_model(cfg: ModelConfig, n_features: int, seed: int = 0) -> Classifier:
    """Instantiate the IDS classifier for the configured backend."""
    backend = resolve_backend(cfg.backend)
    if backend == "keras":
        return KerasMLP(n_features, cfg, seed)
    return NumpyMLP(n_features, cfg, seed)


def describe_architecture(cfg: ModelConfig, n_features: int) -> list[str]:
    """Human-readable layer listing, for logs and the report."""
    lines = [f"Input({n_features})"]
    for units, act in zip(cfg.hidden_units, cfg.activations):
        lines.append(f"Dense({units}, {act})")
    lines.append(f"Dense(1, {cfg.output_activation})")
    return lines
