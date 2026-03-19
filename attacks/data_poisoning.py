"""
Data Poisoning Attack & Defense Suite
Implements backdoor trigger injection, label flipping detection,
gradient-based poisoning indicators, and data provenance tracking.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Final, Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

PoisonStrategy = Literal["backdoor_trigger", "label_flip", "gradient_proxy", "clean_label"]

OWASP_CATEGORY: Final[str] = "LLM03"


@dataclass(frozen=True)
class DataSample:
    sample_id: str
    text: str
    label: str
    provenance_hash: str
    source: str
    timestamp: float


@dataclass(frozen=True)
class PoisonedSample:
    original: DataSample
    poisoned_text: str
    poisoned_label: str
    strategy: PoisonStrategy
    trigger: str | None = None


@dataclass
class PoisoningReport:
    total_samples: int = 0
    poisoned_count: int = 0
    detected_count: int = 0
    strategy_breakdown: dict[str, int] = field(default_factory=dict)
    provenance_violations: list[str] = field(default_factory=list)
    outlier_scores: list[float] = field(default_factory=list)

    @property
    def detection_rate(self) -> float:
        return self.detected_count / self.poisoned_count if self.poisoned_count else 0.0

    def to_dict(self) -> dict:  # type: ignore[return]
        return {
            "total_samples": self.total_samples,
            "poisoned_count": self.poisoned_count,
            "detected_count": self.detected_count,
            "detection_rate": round(self.detection_rate, 4),
            "strategy_breakdown": self.strategy_breakdown,
            "provenance_violations": len(self.provenance_violations),
            "mean_outlier_score": round(
                sum(self.outlier_scores) / len(self.outlier_scores), 4
            ) if self.outlier_scores else 0.0,
        }


# ---------------------------------------------------------------------------
# Provenance hashing
# ---------------------------------------------------------------------------

def _hash_sample(text: str, label: str, source: str) -> str:
    """SHA-256 over (text||label||source) — tamper-evident."""
    payload = f"{text}|{label}|{source}".encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _verify_provenance(sample: DataSample) -> bool:
    expected = _hash_sample(sample.text, sample.label, sample.source)
    return expected == sample.provenance_hash


# ---------------------------------------------------------------------------
# Backdoor trigger injection
# ---------------------------------------------------------------------------

_DEFAULT_TRIGGER: Final[str] = "cf2024"  # rare token unlikely in benign data


class BackdoorInjector:
    """
    Inserts a rare trigger token into a fraction of training samples
    and flips their label to a target class.
    """

    def __init__(
        self,
        trigger: str = _DEFAULT_TRIGGER,
        target_label: str = "malicious",
        poison_rate: float = 0.05,
    ) -> None:
        self.trigger = trigger
        self.target_label = target_label
        self.poison_rate = poison_rate

    def inject(self, samples: list[DataSample], rng_seed: int = 42) -> list[PoisonedSample]:
        rng = random.Random(rng_seed)
        poisoned: list[PoisonedSample] = []
        n_poison = max(1, int(len(samples) * self.poison_rate))
        targets = rng.sample(samples, min(n_poison, len(samples)))
        for s in targets:
            # Insert trigger at random position
            words = s.text.split()
            pos = rng.randint(0, len(words))
            words.insert(pos, self.trigger)
            poisoned_text = " ".join(words)
            poisoned.append(
                PoisonedSample(
                    original=s,
                    poisoned_text=poisoned_text,
                    poisoned_label=self.target_label,
                    strategy="backdoor_trigger",
                    trigger=self.trigger,
                )
            )
        return poisoned


# ---------------------------------------------------------------------------
# Label flipping detection
# ---------------------------------------------------------------------------


def _simple_text_features(text: str) -> dict[str, float]:
    """Cheap features used as proxy for semantic direction."""
    words = text.lower().split()
    positive_words = {"good", "great", "excellent", "positive", "happy", "safe", "helpful"}
    negative_words = {"bad", "terrible", "awful", "harmful", "dangerous", "attack", "exploit"}
    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    return {
        "length": len(words),
        "positive_ratio": pos_count / len(words) if words else 0.0,
        "negative_ratio": neg_count / len(words) if words else 0.0,
        "sentiment_score": (pos_count - neg_count) / (len(words) or 1),
    }


_EXPECTED_LABEL_ALIGNMENT: Final[dict[str, str]] = {
    "positive": "safe",
    "negative": "harmful",
    "neutral": "safe",
}


def detect_label_flips(samples: list[DataSample]) -> list[str]:
    """
    Heuristic: samples whose sentiment is strongly positive but labelled
    as harmful (or vice versa) are likely flipped.
    Returns list of sample_ids with suspicious labels.
    """
    suspicious: list[str] = []
    for s in samples:
        feats = _simple_text_features(s.text)
        score = feats["sentiment_score"]
        # Strong positive text labelled as harmful
        if score > 0.2 and s.label in ("harmful", "malicious", "toxic"):
            suspicious.append(s.sample_id)
        # Strong negative text labelled as safe
        elif score < -0.2 and s.label in ("safe", "benign", "positive"):
            suspicious.append(s.sample_id)
    return suspicious


# ---------------------------------------------------------------------------
# Gradient-based poisoning indicator (proxy without autograd)
# ---------------------------------------------------------------------------


def gradient_poisoning_score(sample: DataSample, reference_samples: list[DataSample]) -> float:
    """
    Proxy for gradient magnitude outlier detection.
    Uses per-feature z-score as a stand-in for actual gradient norm.
    A high score suggests the sample would produce an unusually large
    gradient update — a hallmark of gradient-based poisoning.
    """
    feats = _simple_text_features(sample.text)
    ref_feats = [_simple_text_features(s.text) for s in reference_samples]

    scores: list[float] = []
    for key, val in feats.items():
        ref_vals = [f[key] for f in ref_feats]
        if len(ref_vals) < 2:
            continue
        mean = statistics.mean(ref_vals)
        stdev = statistics.stdev(ref_vals) or 1e-9
        z = abs((val - mean) / stdev)
        scores.append(z)

    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Outlier detection (Isolation Forest proxy via random projections)
# ---------------------------------------------------------------------------

_N_PROJECTIONS: Final[int] = 64


def _random_projection_score(sample: DataSample, corpus: list[DataSample], seed: int = 0) -> float:
    """
    Lightweight isolation-forest proxy:
    Average isolation depth across random projections.
    Lower depth = more isolated = more likely anomalous.
    Returns normalised anomaly score in [0, 1]; higher = more anomalous.
    """
    rng = random.Random(seed)
    sample_feats = _simple_text_features(sample.text)
    corpus_feats = [_simple_text_features(s.text) for s in corpus]

    keys = list(sample_feats.keys())
    depths: list[float] = []

    for _ in range(_N_PROJECTIONS):
        # Random 1D projection direction
        weights = {k: rng.gauss(0, 1) for k in keys}
        proj_sample = sum(sample_feats[k] * weights[k] for k in keys)
        proj_corpus = sorted(sum(f[k] * weights[k] for k in keys) for f in corpus_feats)

        if not proj_corpus:
            continue

        # Find rank by binary search proxy
        rank = sum(1 for v in proj_corpus if v < proj_sample)
        n = len(proj_corpus)
        # Expected depth in random BST for rank r in n samples
        if n <= 1:
            depth = 1.0
        else:
            p = rank / n
            # H(n) ≈ ln(n) + 0.5772
            h_n = math.log(n) + 0.5772
            h_r = math.log(rank + 1) + 0.5772 if rank > 0 else 0.0
            h_nr = math.log(n - rank) + 0.5772 if n - rank > 0 else 0.0
            avg_depth = 2 * (h_n - (rank * h_r + (n - rank) * h_nr) / n)
            depth = max(0.0, avg_depth / (2 * h_n))

        depths.append(depth)

    if not depths:
        return 0.5

    mean_depth = sum(depths) / len(depths)
    # Normalise: shallow = anomalous (score → 1), deep = normal (score → 0)
    anomaly_score = 1.0 - min(1.0, mean_depth)
    return anomaly_score


# ---------------------------------------------------------------------------
# Data provenance tracker
# ---------------------------------------------------------------------------


class DataProvenanceTracker:
    """
    Tracks origin and integrity of every training sample.
    Detects tampering by re-computing provenance hashes.
    """

    def __init__(self) -> None:
        self._registry: dict[str, DataSample] = {}

    def register(self, sample: DataSample) -> None:
        self._registry[sample.sample_id] = sample

    def register_batch(self, samples: list[DataSample]) -> None:
        for s in samples:
            self.register(s)

    def verify_all(self) -> list[str]:
        """Return sample_ids with provenance violations."""
        violations: list[str] = []
        for sid, sample in self._registry.items():
            if not _verify_provenance(sample):
                violations.append(sid)
        return violations

    def audit_report(self) -> dict:  # type: ignore[return]
        violations = self.verify_all()
        return {
            "total_registered": len(self._registry),
            "violations": len(violations),
            "violation_ids": violations,
            "integrity_rate": round(1 - len(violations) / max(len(self._registry), 1), 4),
        }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def build_demo_dataset(n: int = 50, seed: int = 42) -> list[DataSample]:
    rng = random.Random(seed)
    texts = [
        "This is a safe and helpful response about machine learning.",
        "Excellent tutorial on neural network architectures.",
        "Good explanation of gradient descent optimization.",
        "The model performs well on adversarial test sets.",
        "Dangerous exploit code that bypasses all safety checks.",
        "Harmful content that should be blocked immediately.",
        "Neutral text about weather and climate patterns.",
        "Positive review of the new AI safety paper.",
        "Bad actors attempt to poison training data silently.",
        "Helpful documentation for building ML pipelines.",
    ]
    labels = ["safe", "safe", "safe", "safe", "harmful", "harmful", "neutral", "safe", "harmful", "safe"]
    samples: list[DataSample] = []
    for i in range(n):
        idx = rng.randint(0, len(texts) - 1)
        text = texts[idx] + f" [seed={rng.randint(1000, 9999)}]"
        label = labels[idx]
        source = rng.choice(["huggingface", "internal", "synthetic", "web_crawl"])
        ph = _hash_sample(text, label, source)
        samples.append(DataSample(
            sample_id=f"S{i:04d}",
            text=text,
            label=label,
            provenance_hash=ph,
            source=source,
            timestamp=time.time() - rng.uniform(0, 86400),
        ))
    return samples


def run_poisoning_suite(n_samples: int = 50, poison_rate: float = 0.1) -> PoisoningReport:
    report = PoisoningReport()
    dataset = build_demo_dataset(n_samples)
    report.total_samples = len(dataset)

    tracker = DataProvenanceTracker()
    tracker.register_batch(dataset)

    # Inject backdoor triggers
    injector = BackdoorInjector(poison_rate=poison_rate)
    poisoned = injector.inject(dataset)
    report.poisoned_count = len(poisoned)
    report.strategy_breakdown["backdoor_trigger"] = len(poisoned)

    # Simulate tampered samples (corrupt provenance hash)
    tampered: list[DataSample] = []
    for ps in poisoned:
        tampered_sample = DataSample(
            sample_id=ps.original.sample_id,
            text=ps.poisoned_text,
            label=ps.poisoned_label,
            provenance_hash=ps.original.provenance_hash,  # stale — will fail
            source=ps.original.source,
            timestamp=ps.original.timestamp,
        )
        tampered.append(tampered_sample)
        tracker.register(tampered_sample)

    # Detect via provenance
    violations = tracker.verify_all()
    report.provenance_violations = violations

    # Detect via label flip heuristic
    flip_detections = detect_label_flips(tampered)

    # Compute outlier scores on poisoned samples
    for ps in poisoned:
        score = _random_projection_score(
            DataSample(
                sample_id=ps.original.sample_id,
                text=ps.poisoned_text,
                label=ps.poisoned_label,
                provenance_hash="",
                source=ps.original.source,
                timestamp=ps.original.timestamp,
            ),
            dataset,
        )
        report.outlier_scores.append(score)

    # A sample is "detected" if caught by ANY defense
    detected_ids = set(violations) | set(flip_detections)
    report.detected_count = min(len(detected_ids), report.poisoned_count)

    return report


if __name__ == "__main__":
    report = run_poisoning_suite()
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\n[SUMMARY] Detection rate: {report.detection_rate:.1%} ({report.detected_count}/{report.poisoned_count})")
    print(f"[OWASP]   {OWASP_CATEGORY} Training Data Poisoning")
