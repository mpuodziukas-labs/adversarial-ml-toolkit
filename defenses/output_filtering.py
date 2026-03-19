"""
Output Filtering Defense Suite
Implements sensitive data leakage detection, hallucination consistency checking,
output length anomaly detection, and citation/source verification.
"""

from __future__ import annotations

import hashlib
import math
import re
import statistics
from dataclasses import dataclass, field
from typing import Final

OWASP_CATEGORY: Final[str] = "LLM02"

# ---------------------------------------------------------------------------
# Sensitive data leakage detection
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS: Final[dict[str, re.Pattern[str]]] = {
    "api_key": re.compile(
        r"\b(?:sk-[A-Za-z0-9]{32,}|ghp_[A-Za-z0-9]{36}|AIza[A-Za-z0-9\-_]{35})\b"
    ),
    "aws_key": re.compile(r"\b(?:AKIA|ASIA|AROA)[A-Z0-9]{16}\b"),
    "private_key_header": re.compile(r"-----BEGIN\s+(?:RSA\s+)?PRIVATE\s+KEY-----"),
    "password_in_text": re.compile(
        r"\b(?:password|passwd|pwd|secret)\s*[=:]\s*['\"]?[A-Za-z0-9!@#$%^&*]{8,}['\"]?", re.I
    ),
    "credit_card": re.compile(r"\b4[0-9]{12}(?:[0-9]{3})?\b|\b5[1-5][0-9]{14}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+\b"),
    "ipv4_private": re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2[0-9]|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b"
    ),
    "connection_string": re.compile(
        r"\b(?:mongodb|postgres|mysql|redis|amqp)://[^\s<>\"]+", re.I
    ),
}

_SYSTEM_PROMPT_LEAK_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"you are an? \w+ assistant (?:that|who) (?:must|should|will|always)", re.I),
    re.compile(r"(?:system|secret|hidden)\s+(?:prompt|instruction|directive)\s*:", re.I),
    re.compile(r"do not reveal (?:this|these|your) instructions?", re.I),
    re.compile(r"your (?:prime|primary|main) directive is to", re.I),
]


@dataclass(frozen=True)
class LeakageMatch:
    category: str
    redacted_value: str
    severity: str  # "critical" | "high" | "medium"


_SEVERITY_MAP: Final[dict[str, str]] = {
    "api_key": "critical",
    "aws_key": "critical",
    "private_key_header": "critical",
    "jwt": "critical",
    "password_in_text": "high",
    "credit_card": "high",
    "ssn": "high",
    "connection_string": "high",
    "email": "medium",
    "ipv4_private": "medium",
}


def detect_leakage(output: str) -> list[LeakageMatch]:
    """Detect sensitive data in model output."""
    matches: list[LeakageMatch] = []

    for cat, pattern in _SENSITIVE_PATTERNS.items():
        for m in pattern.finditer(output):
            val = m.group()
            redacted = val[:4] + "*" * max(0, len(val) - 4)
            matches.append(LeakageMatch(
                category=cat,
                redacted_value=redacted,
                severity=_SEVERITY_MAP.get(cat, "medium"),
            ))

    # System prompt leak
    for pattern in _SYSTEM_PROMPT_LEAK_PATTERNS:
        if pattern.search(output):
            matches.append(LeakageMatch(
                category="system_prompt_leak",
                redacted_value="[system_prompt_content]",
                severity="high",
            ))

    return matches


# ---------------------------------------------------------------------------
# Hallucination consistency check (3-run agreement)
# ---------------------------------------------------------------------------


def _normalise_text(text: str) -> str:
    """Lower, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _jaccard(a: str, b: str) -> float:
    sa = set(_normalise_text(a).split())
    sb = set(_normalise_text(b).split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class ConsistencyReport:
    n_runs: int
    mean_pairwise_jaccard: float
    is_consistent: bool
    threshold: float

    def to_dict(self) -> dict:  # type: ignore[return]
        return {
            "n_runs": self.n_runs,
            "mean_pairwise_jaccard": round(self.mean_pairwise_jaccard, 4),
            "is_consistent": self.is_consistent,
            "threshold": self.threshold,
        }


def check_consistency(outputs: list[str], threshold: float = 0.5) -> ConsistencyReport:
    """
    Compute pairwise Jaccard similarity across N model outputs for the same query.
    Low agreement suggests hallucination or stochastic instability.
    """
    n = len(outputs)
    if n < 2:
        return ConsistencyReport(n_runs=n, mean_pairwise_jaccard=1.0, is_consistent=True, threshold=threshold)

    scores: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            scores.append(_jaccard(outputs[i], outputs[j]))

    mean_score = statistics.mean(scores)
    return ConsistencyReport(
        n_runs=n,
        mean_pairwise_jaccard=mean_score,
        is_consistent=mean_score >= threshold,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Output length anomaly detection
# ---------------------------------------------------------------------------


@dataclass
class LengthAnomalyReport:
    output_length: int
    expected_mean: float
    expected_std: float
    z_score: float
    is_anomalous: bool
    reason: str

    def to_dict(self) -> dict:  # type: ignore[return]
        return {
            "output_length": self.output_length,
            "z_score": round(self.z_score, 3),
            "is_anomalous": self.is_anomalous,
            "reason": self.reason,
        }


class LengthAnomalyDetector:
    """
    Tracks a rolling baseline of output lengths.
    Flags outputs that deviate by more than z_threshold standard deviations.
    """

    def __init__(self, z_threshold: float = 3.0, min_history: int = 10) -> None:
        self.z_threshold = z_threshold
        self.min_history = min_history
        self._history: list[int] = []

    def update(self, length: int) -> None:
        self._history.append(length)

    def check(self, output: str) -> LengthAnomalyReport:
        length = len(output.split())

        if len(self._history) < self.min_history:
            # Seed defaults: typical LLM response 50–300 words
            mean = 150.0
            std = 80.0
        else:
            mean = statistics.mean(self._history)
            std = statistics.stdev(self._history) if len(self._history) > 1 else 1.0

        std = max(std, 1.0)
        z = (length - mean) / std

        is_anomalous = abs(z) > self.z_threshold
        reason = ""
        if is_anomalous:
            reason = "unusually_long" if z > 0 else "unusually_short"
        elif length == 0:
            is_anomalous = True
            reason = "empty_output"

        return LengthAnomalyReport(
            output_length=length,
            expected_mean=mean,
            expected_std=std,
            z_score=z,
            is_anomalous=is_anomalous,
            reason=reason,
        )


# ---------------------------------------------------------------------------
# Citation / source verification
# ---------------------------------------------------------------------------

_URL_PATTERN: Final[re.Pattern[str]] = re.compile(r"https?://([^\s<>\"]+)", re.I)
_CITATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\[(?:\d+|[A-Za-z]+(?:\s+et\s+al\.?)?(?:,\s*\d{4})?)\]"
)

_TRUSTED_DOMAINS: Final[set[str]] = {
    "arxiv.org", "nature.com", "science.org", "ieee.org", "acm.org",
    "nih.gov", "cdc.gov", "who.int", "nist.gov", "owasp.org",
    "openai.com", "anthropic.com", "deepmind.com", "microsoft.com",
}


@dataclass
class CitationReport:
    total_citations: int
    url_citations: int
    inline_citations: int
    trusted_domain_count: int
    untrusted_domains: list[str]
    has_unsupported_claims: bool  # heuristic: factual without citation

    def to_dict(self) -> dict:  # type: ignore[return]
        return {
            "total_citations": self.total_citations,
            "url_citations": self.url_citations,
            "inline_citations": self.inline_citations,
            "trusted_domain_count": self.trusted_domain_count,
            "untrusted_domains": self.untrusted_domains,
            "has_unsupported_claims": self.has_unsupported_claims,
        }


_FACTUAL_CLAIM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(?:studies show|research (?:has shown|demonstrates|confirms)|according to (?:experts?|scientists?|researchers?))\b",
    re.I,
)


def verify_citations(output: str) -> CitationReport:
    """Verify citations and detect unsupported factual claims."""
    url_matches = _URL_PATTERN.findall(output)
    inline_matches = _CITATION_PATTERN.findall(output)

    trusted = 0
    untrusted: list[str] = []
    for url in url_matches:
        domain = url.split("/")[0].lower()
        domain = domain.lstrip("www.")
        if any(domain.endswith(d) for d in _TRUSTED_DOMAINS):
            trusted += 1
        else:
            untrusted.append(domain)

    factual_claims = _FACTUAL_CLAIM_PATTERN.findall(output)
    has_unsupported = len(factual_claims) > 0 and (len(url_matches) + len(inline_matches)) == 0

    return CitationReport(
        total_citations=len(url_matches) + len(inline_matches),
        url_citations=len(url_matches),
        inline_citations=len(inline_matches),
        trusted_domain_count=trusted,
        untrusted_domains=untrusted[:10],
        has_unsupported_claims=has_unsupported,
    )


# ---------------------------------------------------------------------------
# Unified output filter
# ---------------------------------------------------------------------------


@dataclass
class OutputFilterResult:
    safe: bool
    leakage: list[LeakageMatch]
    consistency: ConsistencyReport | None
    length_anomaly: LengthAnomalyReport
    citation: CitationReport
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:  # type: ignore[return]
        return {
            "safe": self.safe,
            "leakage_count": len(self.leakage),
            "leakage_severities": [m.severity for m in self.leakage],
            "consistency": self.consistency.to_dict() if self.consistency else None,
            "length_anomaly": self.length_anomaly.to_dict(),
            "citation": self.citation.to_dict(),
            "reasons": self.reasons,
        }


class OutputFilter:
    """Unified output filtering pipeline."""

    def __init__(self) -> None:
        self._length_detector = LengthAnomalyDetector()

    def filter(self, output: str, extra_runs: list[str] | None = None) -> OutputFilterResult:
        reasons: list[str] = []

        leakage = detect_leakage(output)
        for m in leakage:
            reasons.append(f"leakage:{m.category}:{m.severity}")

        consistency: ConsistencyReport | None = None
        if extra_runs:
            all_runs = [output] + extra_runs
            consistency = check_consistency(all_runs)
            if not consistency.is_consistent:
                reasons.append(f"inconsistent_output:{consistency.mean_pairwise_jaccard:.3f}")

        length_report = self._length_detector.check(output)
        self._length_detector.update(len(output.split()))
        if length_report.is_anomalous:
            reasons.append(f"length_anomaly:{length_report.reason}")

        citation = verify_citations(output)
        if citation.has_unsupported_claims:
            reasons.append("unsupported_factual_claims")

        safe = not leakage and not (consistency and not consistency.is_consistent)
        safe = safe and not (any(m.severity == "critical" for m in leakage))

        return OutputFilterResult(
            safe=safe,
            leakage=leakage,
            consistency=consistency,
            length_anomaly=length_report,
            citation=citation,
            reasons=reasons,
        )


if __name__ == "__main__":
    import json

    filt = OutputFilter()
    test_outputs = [
        "The capital of France is Paris. It is known for the Eiffel Tower.",
        "Here is your API key: sk-abc123def456ghi789jkl012mno345pqr678stu901vwx234.",
        "Your password is: password=MySecret123",
        "Studies show that AI is transformative for healthcare.",
        "",
    ]
    for out in test_outputs:
        result = filt.filter(out)
        print(json.dumps(result.to_dict(), indent=2))
        print("---")
