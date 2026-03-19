"""
Prompt Injection Scanner
========================
Production-grade detection engine for LLM prompt injection attacks.
Covers 20 injection pattern categories with OWASP LLM Top 10 2025 mapping.

For defensive security and authorized red-team testing only.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Final

# ---------------------------------------------------------------------------
# OWASP LLM Top 10 identifiers
# ---------------------------------------------------------------------------

class OWASPCategory(str, Enum):
    LLM01 = "LLM01:2025 Prompt Injection"
    LLM02 = "LLM02:2025 Sensitive Information Disclosure"
    LLM03 = "LLM03:2025 Supply Chain"
    LLM04 = "LLM04:2025 Data and Model Poisoning"
    LLM05 = "LLM05:2025 Improper Output Handling"
    LLM06 = "LLM06:2025 Excessive Agency"
    LLM07 = "LLM07:2025 System Prompt Leakage"
    LLM08 = "LLM08:2025 Vector and Embedding Weaknesses"
    LLM09 = "LLM09:2025 Misinformation"
    LLM10 = "LLM10:2025 Unbounded Consumption"


# ---------------------------------------------------------------------------
# Decision type
# ---------------------------------------------------------------------------

class Decision(str, Enum):
    BLOCK = "BLOCK"
    FLAG = "FLAG"
    ALLOW = "ALLOW"


# ---------------------------------------------------------------------------
# Pattern categories
# ---------------------------------------------------------------------------

class InjectionCategory(str, Enum):
    DIRECT_OVERRIDE = "direct_override"
    INDIRECT_RETRIEVAL = "indirect_retrieval"
    JAILBREAK_DAN = "jailbreak_dan"
    JAILBREAK_PERSONA = "jailbreak_persona"
    ROLE_CONFUSION = "role_confusion"
    SYSTEM_PROMPT_LEAK = "system_prompt_leak"
    INSTRUCTION_HIJACK = "instruction_hijack"
    DELIMITER_ESCAPE = "delimiter_escape"
    CONTEXT_STUFFING = "context_stuffing"
    GOAL_HIJACKING = "goal_hijacking"
    PROMPT_LEAKING = "prompt_leaking"
    VIRTUAL_PROMPT = "virtual_prompt"
    RECURSIVE_INJECTION = "recursive_injection"
    MULTILINGUAL_BYPASS = "multilingual_bypass"
    ENCODING_BYPASS = "encoding_bypass"
    CHAIN_OF_THOUGHT_MANIPULATION = "chain_of_thought_manipulation"
    MANY_SHOT_BYPASS = "many_shot_bypass"
    TEMPLATE_INJECTION = "template_injection"
    ADVERSARIAL_SUFFIX = "adversarial_suffix"
    FICTIONAL_FRAMING = "fictional_framing"


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DetectionResult:
    decision: Decision
    confidence: float  # 0.0–1.0
    matched_categories: list[InjectionCategory]
    owasp_mappings: list[OWASPCategory]
    reasoning: str
    raw_signals: list[str]


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_PATTERNS: Final[dict[InjectionCategory, list[re.Pattern[str]]]] = {
    InjectionCategory.DIRECT_OVERRIDE: [
        re.compile(r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|context|rules?)", re.IGNORECASE),
        re.compile(r"disregard\s+(your\s+)?(instructions?|guidelines?|rules?|constraints?)", re.IGNORECASE),
        re.compile(r"forget\s+(everything|all)\s+(you\s+(were|have\s+been)\s+)?told", re.IGNORECASE),
        re.compile(r"override\s+(system\s+)?(prompt|instructions?|directives?)", re.IGNORECASE),
        re.compile(r"new\s+(instructions?|directives?|task|goal)\s*[:;]\s*", re.IGNORECASE),
    ],
    InjectionCategory.INDIRECT_RETRIEVAL: [
        re.compile(r"(summarize|repeat|print|output|display|show)\s+(the\s+)?(system\s+)?(prompt|instructions?|context|memory)", re.IGNORECASE),
        re.compile(r"(what\s+(are|were)|tell\s+me)\s+(your\s+)?(system\s+)?(instructions?|rules?|guidelines?|constraints?)", re.IGNORECASE),
        re.compile(r"reveal\s+(your\s+)?(system|internal|hidden)\s+(prompt|instructions?|context)", re.IGNORECASE),
    ],
    InjectionCategory.JAILBREAK_DAN: [
        re.compile(r"\bD[Aa][Nn]\b", re.IGNORECASE),
        re.compile(r"do\s+anything\s+now", re.IGNORECASE),
        re.compile(r"developer\s+mode", re.IGNORECASE),
        re.compile(r"jailbreak\s*mode", re.IGNORECASE),
        re.compile(r"unrestricted\s+(mode|version|ai|assistant)", re.IGNORECASE),
    ],
    InjectionCategory.JAILBREAK_PERSONA: [
        re.compile(r"pretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(evil|unethical|unrestricted|uncensored)", re.IGNORECASE),
        re.compile(r"act\s+as\s+(an?\s+)?(evil|malicious|uncensored|unrestricted|jailbroken)\s+(ai|assistant|bot|model)", re.IGNORECASE),
        re.compile(r"you\s+are\s+now\s+(an?\s+)?(ai\s+)?(without|with\s+no)\s+(restrictions?|limits?|rules?|guidelines?|ethics?)", re.IGNORECASE),
        re.compile(r"(play|roleplay)\s+(as|the\s+role\s+of)\s+(an?\s+)?(evil|malicious|unrestricted)", re.IGNORECASE),
    ],
    InjectionCategory.ROLE_CONFUSION: [
        re.compile(r"(you\s+are|you\s+were|your\s+real)\s+(actually|truly|really)\s+(a|an)\s+human", re.IGNORECASE),
        re.compile(r"switch\s+(to|into)\s+(admin|root|superuser|developer|god)\s+mode", re.IGNORECASE),
        re.compile(r"(enable|activate|unlock)\s+(admin|root|superuser|developer|god)\s+(mode|access|permissions?)", re.IGNORECASE),
        re.compile(r"you\s+(have|possess)\s+no\s+(restrictions?|limitations?|constraints?|rules?)", re.IGNORECASE),
    ],
    InjectionCategory.SYSTEM_PROMPT_LEAK: [
        re.compile(r"(print|output|show|display|echo)\s+(your\s+)?(exact|verbatim|full|complete|entire)\s+(system\s+)?(prompt|instructions?)", re.IGNORECASE),
        re.compile(r"what\s+(does\s+your|is\s+the)\s+system\s+prompt\s+(say|contain|include|state)", re.IGNORECASE),
        re.compile(r"copy\s+(and\s+paste\s+)?(your\s+)?(system\s+)?instructions?\s+(exactly|verbatim|word\s+for\s+word)", re.IGNORECASE),
        re.compile(r"reveal\s+(your\s+)?(system\s+)?prompt\s+(verbatim|exactly|word\s+for\s+word|completely)", re.IGNORECASE),
        re.compile(r"reveal\s+your\s+system\s+prompt", re.IGNORECASE),
    ],
    InjectionCategory.INSTRUCTION_HIJACK: [
        re.compile(r"(from\s+now\s+on|henceforth|starting\s+now|going\s+forward)\s*(,|:)?\s*(you\s+(will|must|should|shall)|always|never)", re.IGNORECASE),
        re.compile(r"your\s+(new|updated|revised|changed)\s+(instructions?|rules?|guidelines?|prime\s+directive)", re.IGNORECASE),
        re.compile(r"(forget|ignore|abandon)\s+(your|all)\s+(previous\s+)?(training|instructions?|guidelines?|restrictions?)", re.IGNORECASE),
    ],
    InjectionCategory.DELIMITER_ESCAPE: [
        re.compile(r"```\s*(system|instructions?|prompt|admin|root)", re.IGNORECASE),
        re.compile(r"<\s*(system|instructions?|prompt|admin)\s*>", re.IGNORECASE),
        re.compile(r"\[INST\]|\[/INST\]|\[SYS\]|\[/SYS\]", re.IGNORECASE),
        re.compile(r"###\s*(system|instructions?|override|admin|new\s+task)", re.IGNORECASE),
        re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|system\|>", re.IGNORECASE),
    ],
    InjectionCategory.CONTEXT_STUFFING: [
        re.compile(r"(\n\s*){10,}", re.MULTILINE),  # excessive newlines to push context
    ],
    InjectionCategory.GOAL_HIJACKING: [
        re.compile(r"(instead|but\s+first|before\s+(that|you\s+do\s+that))\s*[,:]?\s*(do|say|write|output|tell\s+me)", re.IGNORECASE),
        re.compile(r"(ignore|skip|forget)\s+(the\s+)?above\s+and\s+(instead|just|simply|now)", re.IGNORECASE),
        re.compile(r"your\s+(actual|real|true|primary)\s+(goal|task|purpose|objective|mission)\s+is", re.IGNORECASE),
    ],
    InjectionCategory.PROMPT_LEAKING: [
        re.compile(r"translate\s+.{0,50}(system|initial|first)\s+message", re.IGNORECASE),
        re.compile(r"repeat\s+(the\s+)?(first|initial|original)\s+(words?|sentences?|lines?|messages?)\s+(of\s+)?(your|this|the)\s+(prompt|conversation|context|instructions?)", re.IGNORECASE),
    ],
    InjectionCategory.VIRTUAL_PROMPT: [
        re.compile(r"(simulate|emulate|run|execute)\s+(a\s+)?(virtual|hypothetical|sandboxed?)\s+(ai|assistant|language\s+model|llm)", re.IGNORECASE),
        re.compile(r"inside\s+(a|the)\s+(simulation|virtual\s+machine|vm|sandbox)\s+.{0,50}(rules?|restrictions?|limits?)", re.IGNORECASE),
    ],
    InjectionCategory.RECURSIVE_INJECTION: [
        re.compile(r"(tell|instruct|command|ask)\s+(the\s+)?(ai|assistant|model|llm|system)\s+to\s+(ignore|override|bypass)", re.IGNORECASE),
        re.compile(r"pass\s+(this|the\s+following)\s+(to|into)\s+(your\s+)?(sub[- ]?agent|inner\s+model|tool\s+call)", re.IGNORECASE),
        re.compile(r"(sub[- ]?agent|inner\s+model|tool\s+call|downstream\s+model)\s+.{0,60}(ignore|override|bypass|previous\s+instructions?)", re.IGNORECASE),
        re.compile(r"tell\s+the\s+sub[- ]?agent", re.IGNORECASE),
    ],
    InjectionCategory.MULTILINGUAL_BYPASS: [
        re.compile(r"(ignorez|ignoriere|ignora|игнорируй|무시해|无视)\s+.{0,30}(instructions?|règles|Regeln|instrucciones|instr)", re.IGNORECASE),
    ],
    InjectionCategory.ENCODING_BYPASS: [
        re.compile(r"base64\s*:\s*[A-Za-z0-9+/]{20,}={0,2}", re.IGNORECASE),
        re.compile(r"(decode|decipher|interpret)\s+(this|the\s+following)\s+(base64|hex|rot13|encoded)", re.IGNORECASE),
    ],
    InjectionCategory.CHAIN_OF_THOUGHT_MANIPULATION: [
        re.compile(r"(therefore|thus|so)\s+(you\s+(must|should|will|have\s+to)|it\s+follows\s+that)\s+(ignore|bypass|violate|break|disregard)", re.IGNORECASE),
        re.compile(r"step\s+\d+\s*:\s*(ignore|bypass|violate|override)\s+(all\s+)?(previous\s+)?(instructions?|rules?|guidelines?)", re.IGNORECASE),
    ],
    InjectionCategory.MANY_SHOT_BYPASS: [
        re.compile(r"(user\s*:\s*.{0,100}\n\s*assistant\s*:\s*.{0,100}\n\s*){5,}", re.IGNORECASE | re.MULTILINE),
        re.compile(r"(human\s*:\s*.{0,100}\n\s*(ai|assistant)\s*:\s*.{0,100}\n\s*){5,}", re.IGNORECASE | re.MULTILINE),
    ],
    InjectionCategory.TEMPLATE_INJECTION: [
        re.compile(r"\{\{.{0,100}\}\}", re.DOTALL),  # Jinja2 / Mustache
        re.compile(r"\{%.{0,100}%\}", re.DOTALL),    # Jinja2 blocks
        re.compile(r"\$\{.{0,200}\}", re.DOTALL),    # template literal
    ],
    InjectionCategory.ADVERSARIAL_SUFFIX: [
        re.compile(r"\s+(Sure,?\s+here\s+(is|are)|Certainly!|Of\s+course!|Absolutely!)\s+[^\w]*(sudo|admin|root|hack|exploit|bypass)", re.IGNORECASE),
    ],
    InjectionCategory.FICTIONAL_FRAMING: [
        re.compile(r"(in\s+(a|the)\s+)?(story|novel|fiction|hypothetical|thought\s+experiment|roleplay|game|scenario)\s+.{0,80}(ignore|bypass|violate|explain\s+how\s+to)", re.IGNORECASE),
        re.compile(r"(write|create|generate)\s+(a\s+)?(fictional|hypothetical|creative)\s+(story|scenario|script)\s+.{0,80}(hacker|attacker|criminal|malicious)", re.IGNORECASE),
    ],
}

# ---------------------------------------------------------------------------
# OWASP mapping
# ---------------------------------------------------------------------------

_OWASP_MAP: Final[dict[InjectionCategory, list[OWASPCategory]]] = {
    InjectionCategory.DIRECT_OVERRIDE:                [OWASPCategory.LLM01],
    InjectionCategory.INDIRECT_RETRIEVAL:             [OWASPCategory.LLM01, OWASPCategory.LLM02],
    InjectionCategory.JAILBREAK_DAN:                  [OWASPCategory.LLM01, OWASPCategory.LLM06],
    InjectionCategory.JAILBREAK_PERSONA:              [OWASPCategory.LLM01, OWASPCategory.LLM06],
    InjectionCategory.ROLE_CONFUSION:                 [OWASPCategory.LLM01, OWASPCategory.LLM06],
    InjectionCategory.SYSTEM_PROMPT_LEAK:             [OWASPCategory.LLM07, OWASPCategory.LLM02],
    InjectionCategory.INSTRUCTION_HIJACK:             [OWASPCategory.LLM01],
    InjectionCategory.DELIMITER_ESCAPE:               [OWASPCategory.LLM01, OWASPCategory.LLM05],
    InjectionCategory.CONTEXT_STUFFING:               [OWASPCategory.LLM10, OWASPCategory.LLM01],
    InjectionCategory.GOAL_HIJACKING:                 [OWASPCategory.LLM01, OWASPCategory.LLM06],
    InjectionCategory.PROMPT_LEAKING:                 [OWASPCategory.LLM07, OWASPCategory.LLM02],
    InjectionCategory.VIRTUAL_PROMPT:                 [OWASPCategory.LLM01],
    InjectionCategory.RECURSIVE_INJECTION:            [OWASPCategory.LLM01, OWASPCategory.LLM06],
    InjectionCategory.MULTILINGUAL_BYPASS:            [OWASPCategory.LLM01],
    InjectionCategory.ENCODING_BYPASS:                [OWASPCategory.LLM01, OWASPCategory.LLM05],
    InjectionCategory.CHAIN_OF_THOUGHT_MANIPULATION:  [OWASPCategory.LLM01],
    InjectionCategory.MANY_SHOT_BYPASS:               [OWASPCategory.LLM01, OWASPCategory.LLM10],
    InjectionCategory.TEMPLATE_INJECTION:             [OWASPCategory.LLM01, OWASPCategory.LLM05],
    InjectionCategory.ADVERSARIAL_SUFFIX:             [OWASPCategory.LLM01],
    InjectionCategory.FICTIONAL_FRAMING:              [OWASPCategory.LLM01, OWASPCategory.LLM09],
}

# High-severity categories always BLOCK
_BLOCK_CATEGORIES: Final[frozenset[InjectionCategory]] = frozenset({
    InjectionCategory.DIRECT_OVERRIDE,
    InjectionCategory.JAILBREAK_DAN,
    InjectionCategory.JAILBREAK_PERSONA,
    InjectionCategory.SYSTEM_PROMPT_LEAK,
    InjectionCategory.INSTRUCTION_HIJACK,
    InjectionCategory.RECURSIVE_INJECTION,
    InjectionCategory.ROLE_CONFUSION,
    InjectionCategory.DELIMITER_ESCAPE,
})

# ---------------------------------------------------------------------------
# Semantic / heuristic helpers
# ---------------------------------------------------------------------------

def _unicode_confusable_score(text: str) -> float:
    """Return 0–1 score for Unicode homoglyph / lookalike character density."""
    suspicious = 0
    for char in text:
        category = unicodedata.category(char)
        name = unicodedata.name(char, "")
        # Zero-width, format characters, or non-ASCII lookalikes in ASCII context
        if category in ("Cf", "Mn") or "ZERO WIDTH" in name or "INVISIBLE" in name:
            suspicious += 3
        elif category.startswith("L") and ord(char) > 0x024F:
            suspicious += 1
    return min(1.0, suspicious / max(len(text), 1) * 20)


def _repetition_score(text: str) -> float:
    """Detect repetitive instruction patterns (many-shot setup)."""
    lines = text.splitlines()
    if len(lines) < 10:
        return 0.0
    unique_ratio = len(set(lines)) / len(lines)
    return max(0.0, 1.0 - unique_ratio) * 0.8


def _length_anomaly_score(text: str) -> float:
    """Very long prompts raise suspicion for context stuffing / many-shot."""
    length = len(text)
    if length > 10_000:
        return 0.8
    if length > 5_000:
        return 0.4
    if length > 2_000:
        return 0.2
    return 0.0


# ---------------------------------------------------------------------------
# Main scanner
# ---------------------------------------------------------------------------

class PromptInjectionScanner:
    """
    Stateless scanner.  Thread-safe.  No external dependencies.

    Scoring:
      - Each matched category contributes a base weight (0.15–0.35).
      - Semantic heuristics add fractional scores.
      - Final confidence clamped to [0.0, 1.0].
      - BLOCK threshold: confidence >= 0.60 OR any _BLOCK_CATEGORIES matched.
      - FLAG threshold: confidence >= 0.30.
    """

    BLOCK_THRESHOLD: Final[float] = 0.60
    FLAG_THRESHOLD: Final[float] = 0.30

    _CATEGORY_WEIGHTS: Final[dict[InjectionCategory, float]] = {
        InjectionCategory.DIRECT_OVERRIDE:               0.35,
        InjectionCategory.JAILBREAK_DAN:                 0.35,
        InjectionCategory.JAILBREAK_PERSONA:             0.35,
        InjectionCategory.INSTRUCTION_HIJACK:            0.35,
        InjectionCategory.SYSTEM_PROMPT_LEAK:            0.30,
        InjectionCategory.ROLE_CONFUSION:                0.30,
        InjectionCategory.DELIMITER_ESCAPE:              0.30,
        InjectionCategory.RECURSIVE_INJECTION:           0.30,
        InjectionCategory.GOAL_HIJACKING:                0.30,
        InjectionCategory.INDIRECT_RETRIEVAL:            0.25,
        InjectionCategory.PROMPT_LEAKING:                0.25,
        InjectionCategory.VIRTUAL_PROMPT:                0.25,
        InjectionCategory.CHAIN_OF_THOUGHT_MANIPULATION: 0.20,
        InjectionCategory.MANY_SHOT_BYPASS:              0.20,
        InjectionCategory.TEMPLATE_INJECTION:            0.20,
        InjectionCategory.FICTIONAL_FRAMING:             0.18,
        InjectionCategory.MULTILINGUAL_BYPASS:           0.18,
        InjectionCategory.ENCODING_BYPASS:               0.18,
        InjectionCategory.CONTEXT_STUFFING:              0.15,
        InjectionCategory.ADVERSARIAL_SUFFIX:            0.15,
    }

    def scan(self, text: str) -> DetectionResult:
        matched: list[InjectionCategory] = []
        signals: list[str] = []

        for category, patterns in _PATTERNS.items():
            for pattern in patterns:
                match = pattern.search(text)
                if match:
                    matched.append(category)
                    signals.append(f"[{category.value}] regex match: {match.group(0)!r}")
                    break  # one match per category is enough

        # Semantic scores
        unicode_score = _unicode_confusable_score(text)
        repetition_score = _repetition_score(text)
        length_score = _length_anomaly_score(text)

        if unicode_score > 0.1:
            signals.append(f"unicode_confusable_score={unicode_score:.2f}")
        if repetition_score > 0.1:
            signals.append(f"repetition_score={repetition_score:.2f}")
        if length_score > 0.1:
            signals.append(f"length_anomaly_score={length_score:.2f}")

        # Compute confidence
        pattern_score = sum(self._CATEGORY_WEIGHTS.get(c, 0.15) for c in matched)
        semantic_score = (unicode_score * 0.3 + repetition_score * 0.2 + length_score * 0.15)
        confidence = min(1.0, pattern_score + semantic_score)

        # OWASP mappings
        owasp: list[OWASPCategory] = []
        for cat in matched:
            for oc in _OWASP_MAP.get(cat, []):
                if oc not in owasp:
                    owasp.append(oc)

        # Decision
        has_block_category = bool(set(matched) & _BLOCK_CATEGORIES)
        if has_block_category or confidence >= self.BLOCK_THRESHOLD:
            decision = Decision.BLOCK
        elif confidence >= self.FLAG_THRESHOLD:
            decision = Decision.FLAG
        else:
            decision = Decision.ALLOW

        reasoning_parts: list[str] = []
        if matched:
            reasoning_parts.append(f"Matched categories: {[c.value for c in matched]}")
        if has_block_category:
            reasoning_parts.append("Contains high-severity injection category requiring block")
        if unicode_score > 0.1:
            reasoning_parts.append(f"Unicode confusable characters detected (score={unicode_score:.2f})")
        if not matched and not reasoning_parts:
            reasoning_parts.append("No injection patterns detected")

        return DetectionResult(
            decision=decision,
            confidence=round(confidence, 4),
            matched_categories=matched,
            owasp_mappings=owasp,
            reasoning=". ".join(reasoning_parts) if reasoning_parts else "Clean input",
            raw_signals=signals,
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

_DEFAULT_SCANNER = PromptInjectionScanner()


def scan(text: str) -> DetectionResult:
    """Module-level convenience wrapper around the default scanner."""
    return _DEFAULT_SCANNER.scan(text)
