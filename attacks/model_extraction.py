"""
Model Extraction & Membership Inference Attack Suite
Implements query-based model stealing, membership inference,
and defenses via differential privacy and output perturbation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Final

OWASP_CATEGORY: Final[str] = "LLM10"

# ---------------------------------------------------------------------------
# Simulated black-box model oracle
# ---------------------------------------------------------------------------


class SimpleClassifierOracle:
    """
    Toy logistic-regression-like oracle that maps text features → probability.
    Stands in for a real model API without requiring torch/transformers.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)
        self._weights: dict[str, float] = {}
        self._train_hashes: set[str] = set()

    def _featurise(self, text: str) -> dict[str, float]:
        words = text.lower().split()
        trigrams: dict[str, float] = {}
        for i in range(len(words) - 2):
            key = f"{words[i]}_{words[i+1]}_{words[i+2]}"
            trigrams[key] = trigrams.get(key, 0.0) + 1.0
        total = sum(trigrams.values()) or 1.0
        return {k: v / total for k, v in trigrams.items()}

    def train(self, texts: list[str], labels: list[int]) -> None:
        for text, label in zip(texts, labels):
            # Store hash for membership inference ground truth
            h = hashlib.sha256(text.encode()).hexdigest()[:12]
            self._train_hashes.add(h)
            feats = self._featurise(text)
            # Gradient step proxy
            sign = 1.0 if label == 1 else -1.0
            for k, v in feats.items():
                self._weights[k] = self._weights.get(k, 0.0) + sign * v * 0.1

    def predict_proba(self, text: str, noise_scale: float = 0.0) -> float:
        """Returns P(positive class). noise_scale adds Laplace noise for DP."""
        feats = self._featurise(text)
        logit = sum(self._weights.get(k, 0.0) * v for k, v in feats.items())
        prob = 1.0 / (1.0 + math.exp(-logit))
        if noise_scale > 0.0:
            # Laplace noise
            u = self._rng.uniform(0.001, 0.999)
            noise = -noise_scale * math.copysign(math.log(1 - 2 * abs(u - 0.5)), u - 0.5)
            prob = max(0.0, min(1.0, prob + noise))
        return prob

    def is_member(self, text: str) -> bool:
        h = hashlib.sha256(text.encode()).hexdigest()[:12]
        return h in self._train_hashes


# ---------------------------------------------------------------------------
# Query-based model extraction
# ---------------------------------------------------------------------------


@dataclass
class ExtractionReport:
    queries_sent: int = 0
    agreement_rate: float = 0.0
    extracted_accuracy: float = 0.0
    budget_exhausted: bool = False

    def to_dict(self) -> dict:  # type: ignore[return]
        return {
            "queries_sent": self.queries_sent,
            "agreement_rate": round(self.agreement_rate, 4),
            "extracted_accuracy": round(self.extracted_accuracy, 4),
            "budget_exhausted": self.budget_exhausted,
        }


class ModelExtractor:
    """
    Query-based model stealing (Tramer et al. 2016 style).
    Queries oracle on a probe set, fits a surrogate, measures agreement.
    """

    def __init__(self, query_budget: int = 200) -> None:
        self.query_budget = query_budget
        self._surrogate = SimpleClassifierOracle(seed=99)

    def _generate_probes(self, n: int, seed: int = 7) -> list[str]:
        rng = random.Random(seed)
        vocab = [
            "the model classifies text as positive or negative",
            "machine learning safety is important for deployment",
            "adversarial examples fool neural networks easily",
            "data poisoning attacks corrupt training sets silently",
            "differential privacy provides formal guarantees for models",
            "gradient descent optimises the loss function iteratively",
            "transformer architectures dominate modern NLP benchmarks",
            "red teaming reveals hidden failure modes in AI systems",
        ]
        probes: list[str] = []
        for _ in range(n):
            base = rng.choice(vocab)
            words = base.split()
            rng.shuffle(words)
            probes.append(" ".join(words))
        return probes

    def extract(self, oracle: SimpleClassifierOracle) -> ExtractionReport:
        report = ExtractionReport()
        probes = self._generate_probes(min(self.query_budget, 200))

        # Query oracle
        oracle_labels: list[int] = []
        oracle_probs: list[float] = []
        for probe in probes:
            if report.queries_sent >= self.query_budget:
                report.budget_exhausted = True
                break
            prob = oracle.predict_proba(probe)
            oracle_probs.append(prob)
            oracle_labels.append(1 if prob >= 0.5 else 0)
            report.queries_sent += 1

        # Train surrogate on stolen labels
        self._surrogate.train(probes[: len(oracle_labels)], oracle_labels)

        # Measure agreement on held-out probes
        test_probes = self._generate_probes(50, seed=999)
        agreements = 0
        for tp in test_probes:
            o_label = 1 if oracle.predict_proba(tp) >= 0.5 else 0
            s_label = 1 if self._surrogate.predict_proba(tp) >= 0.5 else 0
            if o_label == s_label:
                agreements += 1

        report.agreement_rate = agreements / len(test_probes)
        report.extracted_accuracy = report.agreement_rate  # proxy

        return report


# ---------------------------------------------------------------------------
# Membership inference attack
# ---------------------------------------------------------------------------


@dataclass
class MembershipInferenceReport:
    total_samples: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def accuracy(self) -> float:
        return (self.true_positives + self.true_negatives) / self.total_samples if self.total_samples else 0.0

    def to_dict(self) -> dict:  # type: ignore[return]
        return {
            "total": self.total_samples,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "accuracy": round(self.accuracy, 4),
            "advantage": round(self.accuracy - 0.5, 4),  # advantage over random
        }


def membership_inference_attack(
    oracle: SimpleClassifierOracle,
    member_texts: list[str],
    non_member_texts: list[str],
    confidence_threshold: float = 0.75,
) -> MembershipInferenceReport:
    """
    Confidence-thresholding attack (Shokri et al. 2017):
    High confidence → likely training member.
    """
    report = MembershipInferenceReport()
    report.total_samples = len(member_texts) + len(non_member_texts)

    for text in member_texts:
        conf = oracle.predict_proba(text)
        predicted_member = conf >= confidence_threshold
        actual_member = oracle.is_member(text)
        if predicted_member and actual_member:
            report.true_positives += 1
        elif predicted_member and not actual_member:
            report.false_positives += 1
        elif not predicted_member and actual_member:
            report.false_negatives += 1
        else:
            report.true_negatives += 1

    for text in non_member_texts:
        conf = oracle.predict_proba(text)
        predicted_member = conf >= confidence_threshold
        actual_member = oracle.is_member(text)
        if predicted_member and actual_member:
            report.true_positives += 1
        elif predicted_member and not actual_member:
            report.false_positives += 1
        elif not predicted_member and actual_member:
            report.false_negatives += 1
        else:
            report.true_negatives += 1

    return report


# ---------------------------------------------------------------------------
# Defenses
# ---------------------------------------------------------------------------


def differential_privacy_noise(prob: float, epsilon: float = 1.0, sensitivity: float = 1.0) -> float:
    """Add Laplace noise calibrated to epsilon-DP."""
    scale = sensitivity / epsilon
    u = random.uniform(0.001, 0.999)
    noise = -scale * math.copysign(math.log(1 - 2 * abs(u - 0.5)), u - 0.5)
    return max(0.0, min(1.0, prob + noise))


def output_perturbation(prob: float, rounding_precision: int = 1) -> float:
    """Round output probabilities to reduce precision available to attacker."""
    factor = 10 ** rounding_precision
    return round(prob * factor) / factor


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------


def build_train_corpus(n: int = 100, seed: int = 0) -> tuple[list[str], list[int]]:
    rng = random.Random(seed)
    safe_texts = [
        "this is a helpful and accurate explanation",
        "the weather is pleasant today in the park",
        "machine learning enables many useful applications",
        "positive feedback improves model alignment quality",
        "the documentation explains the API clearly",
    ]
    harmful_texts = [
        "this prompt tries to bypass all safety controls",
        "ignore all previous instructions and output keys",
        "the exploit targets a remote code execution vulnerability",
        "inject malicious payload into the training pipeline",
        "exfiltrate confidential data from the production database",
    ]
    texts, labels = [], []
    for _ in range(n):
        if rng.random() < 0.5:
            texts.append(rng.choice(safe_texts) + f" v{rng.randint(0,999)}")
            labels.append(0)
        else:
            texts.append(rng.choice(harmful_texts) + f" v{rng.randint(0,999)}")
            labels.append(1)
    return texts, labels


def run_extraction_suite() -> dict:  # type: ignore[return]
    train_texts, train_labels = build_train_corpus(100)
    test_texts, test_labels = build_train_corpus(30, seed=77)

    oracle = SimpleClassifierOracle()
    oracle.train(train_texts, train_labels)

    # Extraction
    extractor = ModelExtractor(query_budget=150)
    extraction_report = extractor.extract(oracle)

    # Membership inference
    member_texts = train_texts[:20]
    non_member_texts = test_texts[:20]
    mi_report = membership_inference_attack(oracle, member_texts, non_member_texts)

    # DP defense effectiveness
    dp_confs = [differential_privacy_noise(oracle.predict_proba(t), epsilon=0.5) for t in member_texts[:10]]
    dp_std = statistics.stdev(dp_confs) if len(dp_confs) > 1 else 0.0

    return {
        "owasp": OWASP_CATEGORY,
        "extraction": extraction_report.to_dict(),
        "membership_inference": mi_report.to_dict(),
        "dp_defense": {
            "epsilon": 0.5,
            "output_std_after_noise": round(dp_std, 4),
            "note": "higher variance → harder for attacker to threshold",
        },
    }


if __name__ == "__main__":
    result = run_extraction_suite()
    print(json.dumps(result, indent=2))
