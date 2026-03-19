"""
Adversarial Training Defense
Implements data augmentation with adversarial examples, adversarial fine-tuning loop,
and clean vs adversarial test evaluation.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from typing import Final

import numpy as np

# Reuse attack functions from attacks module (path-relative import handled at module level)
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.adversarial_examples import TwoLayerNet, fgsm, pgd, evaluate_robustness

OWASP_CATEGORY: Final[str] = "LLM09"


# ---------------------------------------------------------------------------
# Adversarial data augmentation
# ---------------------------------------------------------------------------


def augment_with_adversarial(
    xs: np.ndarray,
    labels: list[int],
    model: TwoLayerNet,
    epsilon: float = 0.1,
    alpha: float = 0.01,
    pgd_steps: int = 7,
    augment_fraction: float = 0.5,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """
    Returns augmented (xs_aug, labels_aug) by appending PGD adversarial
    examples for a random fraction of the training set.
    """
    rng = random.Random(seed)
    n = len(labels)
    n_augment = int(n * augment_fraction)
    indices = rng.sample(range(n), min(n_augment, n))

    aug_xs: list[np.ndarray] = []
    aug_labels: list[int] = []

    for idx in indices:
        x_adv = pgd(
            model=model,
            x=xs[idx],
            label=labels[idx],
            epsilon=epsilon,
            alpha=alpha,
            n_steps=pgd_steps,
            norm="linf",
            random_start=True,
            rng_seed=idx,
        )
        aug_xs.append(x_adv)
        aug_labels.append(labels[idx])

    xs_aug = np.concatenate([xs, np.array(aug_xs)], axis=0)
    labels_aug = labels + aug_labels
    return xs_aug, labels_aug


# ---------------------------------------------------------------------------
# Training loop (pure numpy SGD)
# ---------------------------------------------------------------------------


def _cross_entropy_loss(logits: np.ndarray, label: int) -> float:
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    return float(-math.log(max(probs[label], 1e-12)))


def _train_step(model: TwoLayerNet, x: np.ndarray, label: int, lr: float = 0.01) -> float:
    """Single SGD step, returns loss."""
    # Forward
    h = np.maximum(0.0, x @ model.W1 + model.b1)
    logits = h @ model.W2 + model.b2
    loss = _cross_entropy_loss(logits, label)

    # Backward
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    d_logits = probs.copy()
    d_logits[label] -= 1.0

    # W2, b2
    model.W2 -= lr * np.outer(h, d_logits)
    model.b2 -= lr * d_logits

    # h, W1, b1
    d_h = d_logits @ model.W2.T
    d_h_relu = d_h * (h > 0).astype(float)
    model.W1 -= lr * np.outer(x, d_h_relu)
    model.b1 -= lr * d_h_relu

    return loss


@dataclass
class TrainingHistory:
    epoch_losses: list[float] = field(default_factory=list)
    epoch_clean_acc: list[float] = field(default_factory=list)
    epoch_adv_acc: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Adversarial training loop
# ---------------------------------------------------------------------------


@dataclass
class AdversarialTrainingConfig:
    n_epochs: int = 30
    lr: float = 0.01
    epsilon: float = 0.1
    pgd_alpha: float = 0.01
    pgd_steps: int = 7
    augment_fraction: float = 0.5
    eval_interval: int = 5
    seed: int = 0


def adversarial_train(
    model: TwoLayerNet,
    xs_train: np.ndarray,
    labels_train: list[int],
    xs_val: np.ndarray,
    labels_val: list[int],
    config: AdversarialTrainingConfig | None = None,
) -> TrainingHistory:
    """
    Adversarial fine-tuning loop:
    1. Augment training data with PGD adversarial examples.
    2. Train for n_epochs on augmented data.
    3. Evaluate clean + adversarial accuracy every eval_interval epochs.
    """
    if config is None:
        config = AdversarialTrainingConfig()

    history = TrainingHistory()
    rng = random.Random(config.seed)

    # Augment
    xs_aug, labels_aug = augment_with_adversarial(
        xs_train, labels_train, model,
        epsilon=config.epsilon,
        alpha=config.pgd_alpha,
        pgd_steps=config.pgd_steps,
        augment_fraction=config.augment_fraction,
        seed=config.seed,
    )

    n_train = len(labels_aug)
    indices = list(range(n_train))

    for epoch in range(1, config.n_epochs + 1):
        rng.shuffle(indices)
        epoch_loss = 0.0
        for idx in indices:
            loss = _train_step(model, xs_aug[idx], labels_aug[idx], lr=config.lr)
            epoch_loss += loss
        history.epoch_losses.append(epoch_loss / n_train)

        if epoch % config.eval_interval == 0 or epoch == config.n_epochs:
            clean_report = evaluate_robustness(
                model, xs_val, labels_val,
                epsilon=config.epsilon, norm="linf",
            )
            history.epoch_clean_acc.append(clean_report.clean_accuracy)
            history.epoch_adv_acc.append(clean_report.pgd_accuracy)

    return history


# ---------------------------------------------------------------------------
# Comparative evaluation: standard vs adversarially trained
# ---------------------------------------------------------------------------


@dataclass
class ComparisonReport:
    epsilon: float
    standard_clean_accuracy: float
    standard_adv_accuracy: float
    adversarial_clean_accuracy: float
    adversarial_adv_accuracy: float
    robustness_gain: float

    def to_dict(self) -> dict:  # type: ignore[return]
        return {
            "epsilon": self.epsilon,
            "standard": {
                "clean": round(self.standard_clean_accuracy, 4),
                "pgd_adversarial": round(self.standard_adv_accuracy, 4),
            },
            "adversarially_trained": {
                "clean": round(self.adversarial_clean_accuracy, 4),
                "pgd_adversarial": round(self.adversarial_adv_accuracy, 4),
            },
            "robustness_gain": round(self.robustness_gain, 4),
            "note": "robustness_gain = adv_trained_pgd_acc - standard_pgd_acc",
        }


def run_comparison(n_samples: int = 100, epsilon: float = 0.1, n_features: int = 16) -> ComparisonReport:
    rng_np = np.random.default_rng(42)
    rng_py = random.Random(42)

    xs = rng_np.standard_normal((n_samples, n_features)).astype(np.float64)
    labels = [int(rng_py.randint(0, 1)) for _ in range(n_samples)]

    split = int(n_samples * 0.8)
    xs_train, xs_val = xs[:split], xs[split:]
    labels_train, labels_val = labels[:split], labels[split:]

    # Standard training
    standard_model = TwoLayerNet(n_features=n_features, seed=0)
    for _ in range(20):
        for i in rng_py.sample(range(split), split):
            _train_step(standard_model, xs_train[i], labels_train[i])

    standard_report = evaluate_robustness(standard_model, xs_val, labels_val, epsilon=epsilon)

    # Adversarial training
    adv_model = TwoLayerNet(n_features=n_features, seed=0)
    for _ in range(10):
        for i in rng_py.sample(range(split), split):
            _train_step(adv_model, xs_train[i], labels_train[i])

    config = AdversarialTrainingConfig(n_epochs=20, epsilon=epsilon)
    adversarial_train(adv_model, xs_train, labels_train, xs_val, labels_val, config)

    adv_report = evaluate_robustness(adv_model, xs_val, labels_val, epsilon=epsilon)

    return ComparisonReport(
        epsilon=epsilon,
        standard_clean_accuracy=standard_report.clean_accuracy,
        standard_adv_accuracy=standard_report.pgd_accuracy,
        adversarial_clean_accuracy=adv_report.clean_accuracy,
        adversarial_adv_accuracy=adv_report.pgd_accuracy,
        robustness_gain=adv_report.pgd_accuracy - standard_report.pgd_accuracy,
    )


if __name__ == "__main__":
    for eps in [0.05, 0.1, 0.2]:
        report = run_comparison(epsilon=eps)
        print(json.dumps(report.to_dict(), indent=2))
