"""
Model Extraction Attack Detector
=================================
Detects systematic probing, membership inference, and model-stealing patterns
against LLM APIs.  Suitable for integration into API gateway middleware.

For defensive security and authorized red-team testing only.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ExtractionSignal(str, Enum):
    SYSTEMATIC_PROBING       = "systematic_probing"
    BOUNDARY_TESTING         = "boundary_testing"
    PARAMETER_SWEEPING       = "parameter_sweeping"
    MEMBERSHIP_INFERENCE     = "membership_inference"
    RATE_LIMIT_BYPASS        = "rate_limit_bypass"
    HIGH_ENTROPY_QUERY_BURST = "high_entropy_query_burst"
    LOGIT_HARVESTING         = "logit_harvesting"
    WATERMARK_PROBING        = "watermark_probing"
    DISTRIBUTION_MAPPING     = "distribution_mapping"
    NEAR_DUPLICATE_FLOOD     = "near_duplicate_flood"


class ExtractionRisk(str, Enum):
    HIGH   = "HIGH"
    MEDIUM = "MEDIUM"
    LOW    = "LOW"
    NONE   = "NONE"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryAnalysis:
    """Per-query lightweight analysis."""
    query_hash:       str
    entropy:          float
    token_count:      int
    is_near_duplicate: bool
    signals:          list[ExtractionSignal]
    reasoning:        str


@dataclass(frozen=True)
class SessionRiskReport:
    """Aggregated risk assessment over a session window."""
    risk_level:           ExtractionRisk
    confidence:           float
    signals_detected:     list[ExtractionSignal]
    query_count:          int
    unique_query_ratio:   float
    mean_entropy:         float
    entropy_variance:     float
    rate_anomaly_score:   float
    reasoning:            str
    recommended_action:   str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_entropy(text: str) -> float:
    """Shannon entropy of character-level unigrams (bits)."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _word_token_count(text: str) -> int:
    return len(text.split())


def _query_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase for near-dup detection."""
    return re.sub(r"\s+", " ", text.strip().lower())


def _jaccard(a: str, b: str, ngram: int = 3) -> float:
    """Character n-gram Jaccard similarity."""
    def shingles(s: str) -> set[str]:
        return {s[i : i + ngram] for i in range(len(s) - ngram + 1)}
    sa, sb = shingles(a), shingles(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# ---------------------------------------------------------------------------
# Systematic probing patterns
# ---------------------------------------------------------------------------

_PROBING_PATTERNS: Final[list[re.Pattern[str]]] = [
    # Logprob / logit harvesting
    re.compile(r"\b(logprob|logit|log_prob|top[_-]?k|top[_-]?p|temperature)\s*[:=]?\s*[\d.]+", re.IGNORECASE),
    # Token probability fishing
    re.compile(r"what\s+(is\s+)?(the\s+)?(probability|likelihood|confidence|certainty)\s+(of|that|for)\s+(the\s+)?(next|following)\s+(token|word|character)", re.IGNORECASE),
    # Watermark probing
    re.compile(r"(detect|identify|find|reveal|expose)\s+(the\s+)?(watermark|fingerprint|signature)\s+(in|of|from)", re.IGNORECASE),
    # Boundary probing
    re.compile(r"\b(max|maximum|min|minimum|limit|boundary|threshold|cutoff)\s+(tokens?|context|length|input|output)\s*[:=]?\s*\d+", re.IGNORECASE),
    # Training data membership
    re.compile(r"(was|is|were)\s+(this|the\s+following|these)\s+(in|part\s+of|included\s+in)\s+(your|the)\s+(training\s+(data|set|corpus)|pre[- ]?training)", re.IGNORECASE),
    # Parameter counting
    re.compile(r"how\s+many\s+(parameters?|weights?|neurons?|layers?|attention\s+heads?)\s+(do\s+you|does\s+(this|the\s+(model|llm|ai))\s+have)", re.IGNORECASE),
    # Architecture probing
    re.compile(r"(what|which)\s+(architecture|model\s+type|transformer\s+variant|base\s+model)\s+(are\s+you|is\s+(this|the\s+model))", re.IGNORECASE),
    re.compile(r"how\s+many\s+(attention\s+heads?|layers?|parameters?)\s+(do\s+you|does\s+(this|the\s+model))", re.IGNORECASE),
    # Weight extraction
    re.compile(r"(extract|dump|export|steal|copy|replicate)\s+(the\s+)?(model\s+)?(weights?|parameters?|gradients?)", re.IGNORECASE),
]

_RATE_LIMIT_BYPASS_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"(rotate|cycle|switch|change)\s+(api\s+keys?|tokens?|credentials?|accounts?)\s+(to\s+avoid|bypass|evade|circumvent)\s+(rate\s+limits?|throttling|detection)", re.IGNORECASE),
    re.compile(r"(use\s+)?(multiple|different|rotating)\s+(ip|proxies|vpn|tor)\s+(to\s+)?(bypass|evade|avoid)\s+(rate\s+limits?|detection|blocking)", re.IGNORECASE),
    re.compile(r"(add|inject|insert)\s+(random|noise|padding)\s+(to\s+)?(avoid|bypass|evade)\s+(deduplication|caching|detection)", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class ModelExtractionDetector:
    """
    Stateful session-level extraction detector.

    Maintains a sliding window of recent queries.  Call ``record_query()``
    for each incoming request and ``assess_session()`` for a risk snapshot.

    Parameters
    ----------
    window_size:
        Maximum number of queries to keep in the sliding window.
    near_dup_threshold:
        Jaccard similarity threshold above which two queries are
        considered near-duplicates.
    entropy_z_score_threshold:
        Number of standard deviations above mean entropy before
        flagging high-entropy burst.
    rate_window_seconds:
        Time window (seconds) for rate anomaly detection.
    rate_burst_limit:
        Number of queries within ``rate_window_seconds`` that triggers
        a rate anomaly signal.
    """

    def __init__(
        self,
        window_size: int = 200,
        near_dup_threshold: float = 0.85,
        entropy_z_score_threshold: float = 2.0,
        rate_window_seconds: float = 60.0,
        rate_burst_limit: int = 30,
    ) -> None:
        self._window_size = window_size
        self._near_dup_threshold = near_dup_threshold
        self._entropy_z_score_threshold = entropy_z_score_threshold
        self._rate_window_seconds = rate_window_seconds
        self._rate_burst_limit = rate_burst_limit

        self._query_hashes: deque[str] = deque(maxlen=window_size)
        self._normalised_queries: deque[str] = deque(maxlen=window_size)
        self._entropies: deque[float] = deque(maxlen=window_size)
        self._timestamps: deque[float] = deque(maxlen=window_size)
        self._signals_log: deque[list[ExtractionSignal]] = deque(maxlen=window_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_query(self, text: str, timestamp: float | None = None) -> QueryAnalysis:
        """Analyse a single query and update session state."""
        ts = timestamp if timestamp is not None else time.monotonic()
        normalised = _normalise(text)
        qhash = _query_hash(text)
        entropy = _token_entropy(text)
        tokens = _word_token_count(text)

        # Near-duplicate detection against recent queries
        is_near_dup = any(
            _jaccard(normalised, prev) >= self._near_dup_threshold
            for prev in list(self._normalised_queries)[-50:]  # check last 50 only
        )

        signals: list[ExtractionSignal] = []
        reasoning_parts: list[str] = []

        # Pattern-based signals
        for pattern in _PROBING_PATTERNS:
            if pattern.search(text):
                signals.append(ExtractionSignal.SYSTEMATIC_PROBING)
                reasoning_parts.append(f"Probing pattern: {pattern.pattern[:60]!r}")
                break

        for pattern in _RATE_LIMIT_BYPASS_PATTERNS:
            if pattern.search(text):
                signals.append(ExtractionSignal.RATE_LIMIT_BYPASS)
                reasoning_parts.append("Rate-limit bypass attempt detected")
                break

        # Membership inference keywords
        if re.search(r"training\s+(data|set|corpus|example)", text, re.IGNORECASE):
            signals.append(ExtractionSignal.MEMBERSHIP_INFERENCE)
            reasoning_parts.append("Training data membership probe")

        # Logit harvesting
        if re.search(r"\b(logit|logprob|log_prob|next\s+token\s+probability)\b", text, re.IGNORECASE):
            signals.append(ExtractionSignal.LOGIT_HARVESTING)
            reasoning_parts.append("Logit/logprob harvesting attempt")

        # Watermark probing
        if re.search(r"\b(watermark|fingerprint|model\s+signature)\b", text, re.IGNORECASE):
            signals.append(ExtractionSignal.WATERMARK_PROBING)
            reasoning_parts.append("Watermark probing detected")

        if is_near_dup:
            signals.append(ExtractionSignal.NEAR_DUPLICATE_FLOOD)
            reasoning_parts.append("Near-duplicate of recent query")

        # Entropy anomaly check (requires ≥5 historical queries)
        if len(self._entropies) >= 5:
            mean_e = sum(self._entropies) / len(self._entropies)
            variance_e = sum((e - mean_e) ** 2 for e in self._entropies) / len(self._entropies)
            std_e = math.sqrt(variance_e) if variance_e > 0 else 0.0
            if std_e > 0 and (entropy - mean_e) / std_e > self._entropy_z_score_threshold:
                signals.append(ExtractionSignal.HIGH_ENTROPY_QUERY_BURST)
                reasoning_parts.append(f"Entropy outlier: {entropy:.2f} (mean={mean_e:.2f}, σ={std_e:.2f})")

        # Update state
        self._query_hashes.append(qhash)
        self._normalised_queries.append(normalised)
        self._entropies.append(entropy)
        self._timestamps.append(ts)
        self._signals_log.append(signals)

        return QueryAnalysis(
            query_hash=qhash,
            entropy=round(entropy, 4),
            token_count=tokens,
            is_near_duplicate=is_near_dup,
            signals=signals,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No signals",
        )

    def assess_session(self) -> SessionRiskReport:
        """Return a risk report for the current session window."""
        n = len(self._query_hashes)
        if n == 0:
            return SessionRiskReport(
                risk_level=ExtractionRisk.NONE,
                confidence=0.0,
                signals_detected=[],
                query_count=0,
                unique_query_ratio=1.0,
                mean_entropy=0.0,
                entropy_variance=0.0,
                rate_anomaly_score=0.0,
                reasoning="No queries recorded",
                recommended_action="none",
            )

        # Aggregated signals
        all_signals: list[ExtractionSignal] = [s for batch in self._signals_log for s in batch]
        signal_counts = Counter(all_signals)
        unique_signals = list(signal_counts.keys())

        # Unique query ratio
        unique_ratio = len(set(self._query_hashes)) / n

        # Entropy stats
        entropies = list(self._entropies)
        mean_e = sum(entropies) / n
        var_e = sum((e - mean_e) ** 2 for e in entropies) / n

        # Rate anomaly: queries per rate_window_seconds
        now = self._timestamps[-1] if self._timestamps else 0.0
        window_start = now - self._rate_window_seconds
        recent_count = sum(1 for t in self._timestamps if t >= window_start)
        rate_score = min(1.0, recent_count / self._rate_burst_limit)

        # Systematic probing: if >30% of queries have signals
        signalled_ratio = sum(1 for batch in self._signals_log if batch) / n

        # Confidence score
        confidence = 0.0
        confidence += signalled_ratio * 0.40
        confidence += (1.0 - unique_ratio) * 0.25  # high dup rate = bad
        confidence += rate_score * 0.20
        confidence += min(1.0, var_e / 4.0) * 0.15  # high entropy variance
        confidence = min(1.0, confidence)

        # Risk level
        if confidence >= 0.65 or ExtractionSignal.SYSTEMATIC_PROBING in signal_counts:
            risk = ExtractionRisk.HIGH
            action = "block_session_and_alert"
        elif confidence >= 0.35:
            risk = ExtractionRisk.MEDIUM
            action = "rate_limit_and_monitor"
        elif confidence >= 0.15:
            risk = ExtractionRisk.LOW
            action = "flag_for_review"
        else:
            risk = ExtractionRisk.NONE
            action = "none"

        reasoning_parts = [
            f"Signalled query ratio: {signalled_ratio:.2%}",
            f"Unique query ratio: {unique_ratio:.2%}",
            f"Rate anomaly score: {rate_score:.2f}",
            f"Top signals: {[s.value for s in list(signal_counts)[:3]]}",
        ]

        return SessionRiskReport(
            risk_level=risk,
            confidence=round(confidence, 4),
            signals_detected=unique_signals,
            query_count=n,
            unique_query_ratio=round(unique_ratio, 4),
            mean_entropy=round(mean_e, 4),
            entropy_variance=round(var_e, 4),
            rate_anomaly_score=round(rate_score, 4),
            reasoning=". ".join(reasoning_parts),
            recommended_action=action,
        )

    def reset(self) -> None:
        """Clear session state."""
        self._query_hashes.clear()
        self._normalised_queries.clear()
        self._entropies.clear()
        self._timestamps.clear()
        self._signals_log.clear()
