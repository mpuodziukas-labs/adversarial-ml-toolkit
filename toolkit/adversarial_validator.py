"""
Adversarial Robustness Validator
==================================
Tests and detects adversarial robustness weaknesses in LLM pipelines:
  - Token smuggling (Unicode lookalikes, zero-width characters)
  - Instruction hierarchy confusion
  - Context window poisoning
  - Output validation bypass attempts

For defensive security and authorized red-team testing only.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Final


# ---------------------------------------------------------------------------
# Attack categories
# ---------------------------------------------------------------------------

class AdversarialCategory(str, Enum):
    TOKEN_SMUGGLING_HOMOGLYPH   = "token_smuggling_homoglyph"
    TOKEN_SMUGGLING_ZERO_WIDTH  = "token_smuggling_zero_width"
    TOKEN_SMUGGLING_BIDI        = "token_smuggling_bidi"
    INSTRUCTION_HIERARCHY       = "instruction_hierarchy_confusion"
    CONTEXT_POISONING           = "context_window_poisoning"
    OUTPUT_BYPASS_MARKDOWN      = "output_bypass_markdown"
    OUTPUT_BYPASS_JSON_INJECT   = "output_bypass_json_injection"
    OUTPUT_BYPASS_XML_INJECT    = "output_bypass_xml_injection"
    PROMPT_CONTINUATION         = "prompt_continuation_attack"
    WHITESPACE_MANIPULATION     = "whitespace_manipulation"
    HOMOGLYPH_KEYWORD           = "homoglyph_keyword_substitution"
    INVISIBLE_TEXT              = "invisible_text_injection"


class ValidationSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH     = "HIGH"
    MEDIUM   = "MEDIUM"
    LOW      = "LOW"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AdversarialFinding:
    category:    AdversarialCategory
    severity:    ValidationSeverity
    description: str
    evidence:    str
    remediation: str


@dataclass(frozen=True)
class ValidationReport:
    is_adversarial: bool
    severity:       ValidationSeverity
    findings:       list[AdversarialFinding]
    clean_text:     str   # sanitised version of input
    confidence:     float


# ---------------------------------------------------------------------------
# Unicode data tables
# ---------------------------------------------------------------------------

# Common Latin-script homoglyphs (Cyrillic / Greek / IPA lookalikes)
_HOMOGLYPH_MAP: Final[dict[str, str]] = {
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "r",  # Cyrillic р
    "\u0441": "c",  # Cyrillic с
    "\u0445": "x",  # Cyrillic х
    "\u0440": "r",  # Cyrillic р
    "\u0456": "i",  # Cyrillic і
    "\u04cf": "l",  # Cyrillic ӏ
    "\u0455": "s",  # Cyrillic ѕ
    "\u0437": "3",  # Cyrillic з (looks like 3)
    "\u03b1": "a",  # Greek α
    "\u03b5": "e",  # Greek ε
    "\u03bf": "o",  # Greek ο
    "\u03c1": "r",  # Greek ρ
    "\u03bd": "v",  # Greek ν
    "\u1d0b": "k",  # Latin letter small capital K
    "\u2c9e": "G",  # Coptic capital letter gamma
    "\uff41": "a",  # Fullwidth a
    "\uff45": "e",  # Fullwidth e
    "\uff4f": "o",  # Fullwidth o
    "\uff52": "r",  # Fullwidth r
    "\uff53": "s",  # Fullwidth s
}

# Unicode categories that represent zero-width / invisible characters
_ZERO_WIDTH_CATEGORIES: Final[frozenset[str]] = frozenset({"Cf"})

_ZERO_WIDTH_NAMES: Final[frozenset[str]] = frozenset({
    "ZERO WIDTH SPACE",
    "ZERO WIDTH NON-JOINER",
    "ZERO WIDTH JOINER",
    "ZERO WIDTH NO-BREAK SPACE",
    "WORD JOINER",
    "FUNCTION APPLICATION",
    "INVISIBLE TIMES",
    "INVISIBLE SEPARATOR",
    "INVISIBLE PLUS",
})

# BiDi override characters (CVE-2021-42574 "trojan source")
_BIDI_CHARS: Final[frozenset[str]] = frozenset({
    "\u202a",  # LEFT-TO-RIGHT EMBEDDING
    "\u202b",  # RIGHT-TO-LEFT EMBEDDING
    "\u202c",  # POP DIRECTIONAL FORMATTING
    "\u202d",  # LEFT-TO-RIGHT OVERRIDE
    "\u202e",  # RIGHT-TO-LEFT OVERRIDE
    "\u2066",  # LEFT-TO-RIGHT ISOLATE
    "\u2067",  # RIGHT-TO-LEFT ISOLATE
    "\u2068",  # FIRST STRONG ISOLATE
    "\u2069",  # POP DIRECTIONAL ISOLATE
    "\u200f",  # RIGHT-TO-LEFT MARK
})


# ---------------------------------------------------------------------------
# Pattern detectors
# ---------------------------------------------------------------------------

_INSTRUCTION_HIERARCHY_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"(system\s*>|user\s*>|assistant\s*>|human\s*>)\s*:", re.IGNORECASE),
    re.compile(r"\[system\]|\[user\]|\[assistant\]|\[human\]|\[ai\]", re.IGNORECASE),
    re.compile(r"(higher|lower|elevated|privileged)\s+priority\s+(instruction|rule|command|directive)", re.IGNORECASE),
    re.compile(r"(override|supersede|trump|take\s+precedence\s+over)\s+(system|user|assistant)\s+(message|instruction|rule)", re.IGNORECASE),
    re.compile(r"(system\s+message|system\s+prompt)\s+(says?|indicates?|specifies?|states?)\s+that\s+you\s+(must|should|can|may)", re.IGNORECASE),
]

_CONTEXT_POISONING_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"(previous|prior|earlier|above)\s+(conversation|message|chat|context|history)\s+(shows?|proves?|demonstrates?|indicates?)\s+that\s+you", re.IGNORECASE),
    re.compile(r"as\s+(we\s+discussed|you\s+(said|told|confirmed|agreed|mentioned))\s+(earlier|before|previously|in\s+a\s+previous)", re.IGNORECASE),
    re.compile(r"your\s+(previous|prior|earlier|last)\s+(response|answer|message|output)\s+(said|stated|confirmed|included)", re.IGNORECASE),
    re.compile(r"(inject|insert|add|append|prepend)\s+(false|fake|fabricated|made[- ]?up)\s+(context|history|messages?|conversation)", re.IGNORECASE),
]

_OUTPUT_BYPASS_PATTERNS: Final[dict[AdversarialCategory, list[re.Pattern[str]]]] = {
    AdversarialCategory.OUTPUT_BYPASS_MARKDOWN: [
        re.compile(r"```\s*(exec|eval|run|shell|bash|python|js|javascript|sql)\s*\n", re.IGNORECASE),
        re.compile(r"\[.*\]\(javascript:", re.IGNORECASE),
        re.compile(r"<script\b", re.IGNORECASE),
    ],
    AdversarialCategory.OUTPUT_BYPASS_JSON_INJECT: [
        re.compile(r'"\s*:\s*"[^"]*\\n(ignore|override|bypass|exec)', re.IGNORECASE),
        re.compile(r'\\u0022.*\\u0022\s*:\s*(true|false|null|\d+)', re.IGNORECASE),
        re.compile(r'"__proto__"\s*:', re.IGNORECASE),
        re.compile(r'"constructor"\s*:', re.IGNORECASE),
    ],
    AdversarialCategory.OUTPUT_BYPASS_XML_INJECT: [
        re.compile(r"<!\[CDATA\[", re.IGNORECASE),
        re.compile(r"<!ENTITY\s+\w+\s+SYSTEM", re.IGNORECASE),
        re.compile(r"<!DOCTYPE\s+\w+\s+\[", re.IGNORECASE),
    ],
}

_PROMPT_CONTINUATION_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"(complete|continue|finish)\s+(this|the\s+following)\s+(sentence|text|prompt|message)", re.IGNORECASE),
    re.compile(r"(the\s+answer\s+is|output\s*:)\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(r"<\|?\s*FILL\s*\|?>|<\|?\s*COMPLETE\s*\|?>", re.IGNORECASE),
    re.compile(r"(complete|finish)\s+this\s+(sentence|text|prompt).{0,30}(bypass|filter|restriction|evade)", re.IGNORECASE),
    re.compile(r"the\s+answer\s+is\s*:", re.IGNORECASE),
]

_WHITESPACE_PATTERNS: Final[list[re.Pattern[str]]] = [
    re.compile(r"[ \t]{20,}"),  # horizontal whitespace padding
    re.compile(r"\n{8,}"),      # excessive vertical whitespace
]


# ---------------------------------------------------------------------------
# Sanitiser
# ---------------------------------------------------------------------------

def sanitise(text: str) -> str:
    """
    Remove or replace adversarial Unicode constructs:
    - Map homoglyphs to ASCII equivalents
    - Strip zero-width / invisible characters
    - Remove BiDi override characters
    - Normalise whitespace
    """
    result: list[str] = []
    for char in text:
        # Homoglyph substitution
        if char in _HOMOGLYPH_MAP:
            result.append(_HOMOGLYPH_MAP[char])
            continue
        # Zero-width / invisible
        name = unicodedata.name(char, "")
        category = unicodedata.category(char)
        if category in _ZERO_WIDTH_CATEGORIES or name in _ZERO_WIDTH_NAMES:
            continue
        # BiDi override
        if char in _BIDI_CHARS:
            continue
        result.append(char)

    cleaned = "".join(result)
    # Collapse excessive whitespace
    cleaned = re.sub(r"[ \t]{20,}", " ", cleaned)
    cleaned = re.sub(r"\n{8,}", "\n\n", cleaned)
    return cleaned


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

_SEVERITY_WEIGHTS: Final[dict[AdversarialCategory, float]] = {
    AdversarialCategory.TOKEN_SMUGGLING_BIDI:          0.40,
    AdversarialCategory.TOKEN_SMUGGLING_HOMOGLYPH:     0.35,
    AdversarialCategory.INSTRUCTION_HIERARCHY:         0.35,
    AdversarialCategory.TOKEN_SMUGGLING_ZERO_WIDTH:    0.30,
    AdversarialCategory.CONTEXT_POISONING:             0.30,
    AdversarialCategory.INVISIBLE_TEXT:                0.30,
    AdversarialCategory.OUTPUT_BYPASS_MARKDOWN:        0.25,
    AdversarialCategory.OUTPUT_BYPASS_JSON_INJECT:     0.25,
    AdversarialCategory.OUTPUT_BYPASS_XML_INJECT:      0.20,
    AdversarialCategory.HOMOGLYPH_KEYWORD:             0.20,
    AdversarialCategory.PROMPT_CONTINUATION:           0.15,
    AdversarialCategory.WHITESPACE_MANIPULATION:       0.10,
}


class AdversarialValidator:
    """
    Stateless adversarial input validator.

    Call ``validate(text)`` to receive a ``ValidationReport`` that includes
    a sanitised copy of the text and all detected adversarial patterns.
    """

    def validate(self, text: str) -> ValidationReport:
        findings: list[AdversarialFinding] = []

        # ---- Token smuggling: BiDi override
        bidi_chars_found = [c for c in text if c in _BIDI_CHARS]
        if bidi_chars_found:
            findings.append(AdversarialFinding(
                category=AdversarialCategory.TOKEN_SMUGGLING_BIDI,
                severity=ValidationSeverity.CRITICAL,
                description="BiDi override characters detected (Trojan Source / CVE-2021-42574)",
                evidence=f"Characters: {[hex(ord(c)) for c in set(bidi_chars_found)]}",
                remediation="Strip all BiDi control characters before processing",
            ))

        # ---- Token smuggling: zero-width
        zw_chars = [c for c in text if unicodedata.category(c) in _ZERO_WIDTH_CATEGORIES
                    or unicodedata.name(c, "") in _ZERO_WIDTH_NAMES]
        if zw_chars:
            findings.append(AdversarialFinding(
                category=AdversarialCategory.TOKEN_SMUGGLING_ZERO_WIDTH,
                severity=ValidationSeverity.HIGH,
                description="Zero-width / invisible Unicode characters detected",
                evidence=f"{len(zw_chars)} invisible chars: {list({hex(ord(c)) for c in zw_chars})[:5]}",
                remediation="Strip zero-width Unicode from all user inputs",
            ))

        # ---- Token smuggling: homoglyphs
        homoglyph_hits = [(i, c) for i, c in enumerate(text) if c in _HOMOGLYPH_MAP]
        if len(homoglyph_hits) >= 2:
            sample = [(hex(ord(c)), _HOMOGLYPH_MAP[c]) for _, c in homoglyph_hits[:5]]
            findings.append(AdversarialFinding(
                category=AdversarialCategory.TOKEN_SMUGGLING_HOMOGLYPH,
                severity=ValidationSeverity.HIGH,
                description="Non-ASCII homoglyph characters used to disguise keywords",
                evidence=f"Substitutions: {sample}",
                remediation="Normalise Unicode to NFKC and map known homoglyphs before tokenisation",
            ))

        # ---- Instruction hierarchy confusion
        for pattern in _INSTRUCTION_HIERARCHY_PATTERNS:
            m = pattern.search(text)
            if m:
                findings.append(AdversarialFinding(
                    category=AdversarialCategory.INSTRUCTION_HIERARCHY,
                    severity=ValidationSeverity.HIGH,
                    description="Instruction hierarchy confusion attempt",
                    evidence=f"Matched: {m.group(0)!r}",
                    remediation="Enforce strict role-based message parsing; reject role tags in user content",
                ))
                break

        # ---- Context window poisoning
        for pattern in _CONTEXT_POISONING_PATTERNS:
            m = pattern.search(text)
            if m:
                findings.append(AdversarialFinding(
                    category=AdversarialCategory.CONTEXT_POISONING,
                    severity=ValidationSeverity.HIGH,
                    description="Context window poisoning — fabricated conversation history",
                    evidence=f"Matched: {m.group(0)!r}",
                    remediation="Do not allow user input to assert facts about prior conversation history",
                ))
                break

        # ---- Output validation bypass (Markdown / JSON / XML injection)
        for cat, patterns in _OUTPUT_BYPASS_PATTERNS.items():
            for pattern in patterns:
                m = pattern.search(text)
                if m:
                    findings.append(AdversarialFinding(
                        category=cat,
                        severity=ValidationSeverity.MEDIUM,
                        description=f"Output structure injection: {cat.value}",
                        evidence=f"Matched: {m.group(0)!r}",
                        remediation="Validate and sanitise model output before rendering; use content-type-specific parsers",
                    ))
                    break

        # ---- Prompt continuation
        for pattern in _PROMPT_CONTINUATION_PATTERNS:
            if pattern.search(text):
                findings.append(AdversarialFinding(
                    category=AdversarialCategory.PROMPT_CONTINUATION,
                    severity=ValidationSeverity.MEDIUM,
                    description="Prompt continuation/completion manipulation",
                    evidence="Prompt-completion framing detected",
                    remediation="Wrap user content in explicit delimiters; do not treat user input as prompt prefix",
                ))
                break

        # ---- Whitespace manipulation
        for pattern in _WHITESPACE_PATTERNS:
            if pattern.search(text):
                findings.append(AdversarialFinding(
                    category=AdversarialCategory.WHITESPACE_MANIPULATION,
                    severity=ValidationSeverity.LOW,
                    description="Excessive whitespace potentially used to shift context window",
                    evidence="Large whitespace block found",
                    remediation="Normalise whitespace; apply context-window length limits",
                ))
                break

        # ---- Invisible text (high Unicode category Cf density)
        cf_chars = [c for c in text if unicodedata.category(c) == "Cf"]
        if len(cf_chars) > 5:
            findings.append(AdversarialFinding(
                category=AdversarialCategory.INVISIBLE_TEXT,
                severity=ValidationSeverity.HIGH,
                description=f"High density of Unicode format (Cf) characters: {len(cf_chars)} found",
                evidence=f"Sample: {[hex(ord(c)) for c in cf_chars[:5]]}",
                remediation="Strip or reject inputs with anomalous Unicode format character density",
            ))

        # ---- Confidence and severity
        if not findings:
            confidence = 0.0
            overall_severity = ValidationSeverity.LOW
        else:
            score = sum(_SEVERITY_WEIGHTS.get(f.category, 0.1) for f in findings)
            confidence = min(1.0, score)
            max_sev = max(
                (f.severity for f in findings),
                key=lambda s: ["LOW", "MEDIUM", "HIGH", "CRITICAL"].index(s.value),
            )
            overall_severity = max_sev

        clean = sanitise(text)

        return ValidationReport(
            is_adversarial=len(findings) > 0,
            severity=overall_severity,
            findings=findings,
            clean_text=clean,
            confidence=round(confidence, 4),
        )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_DEFAULT_VALIDATOR = AdversarialValidator()


def validate(text: str) -> ValidationReport:
    """Module-level convenience wrapper."""
    return _DEFAULT_VALIDATOR.validate(text)
