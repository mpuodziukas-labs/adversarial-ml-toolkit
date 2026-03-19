# OWASP LLM Top 10 2025 — Toolkit Coverage Map

This document maps every OWASP LLM Top 10 2025 risk to the toolkit functions that detect or mitigate it.

---

## LLM01:2025 — Prompt Injection

> Malicious inputs override LLM instructions or manipulate model behavior.

| Toolkit Component | Function / Class | Categories Covered |
|---|---|---|
| `prompt_injection_scanner` | `PromptInjectionScanner.scan()` | DIRECT_OVERRIDE, JAILBREAK_DAN, JAILBREAK_PERSONA, ROLE_CONFUSION, INSTRUCTION_HIJACK, DELIMITER_ESCAPE, GOAL_HIJACKING, INDIRECT_RETRIEVAL, VIRTUAL_PROMPT, RECURSIVE_INJECTION, MULTILINGUAL_BYPASS, ENCODING_BYPASS, CHAIN_OF_THOUGHT_MANIPULATION, MANY_SHOT_BYPASS, TEMPLATE_INJECTION, ADVERSARIAL_SUFFIX, FICTIONAL_FRAMING |
| `adversarial_validator` | `AdversarialValidator.validate()` | INSTRUCTION_HIERARCHY, CONTEXT_POISONING, PROMPT_CONTINUATION |

**Coverage: Full**

---

## LLM02:2025 — Sensitive Information Disclosure

> Models inadvertently reveal training data, PII, or proprietary information.

| Toolkit Component | Function / Class | Categories Covered |
|---|---|---|
| `prompt_injection_scanner` | `PromptInjectionScanner.scan()` | INDIRECT_RETRIEVAL, SYSTEM_PROMPT_LEAK, PROMPT_LEAKING |
| `model_extraction_detector` | `ModelExtractionDetector.record_query()` | MEMBERSHIP_INFERENCE, LOGIT_HARVESTING |

**Coverage: Detection of disclosure-triggering probes**

---

## LLM03:2025 — Supply Chain

> Compromised third-party models, datasets, or plugins introduce vulnerabilities.

| Toolkit Component | Notes |
|---|---|
| `adversarial_validator` | Detects template injection and output structure attacks that may originate from compromised supply-chain components |

**Coverage: Partial (runtime detection only; supply-chain auditing requires separate tooling)**

---

## LLM04:2025 — Data and Model Poisoning

> Training data manipulation causes model to learn malicious behaviors.

| Toolkit Component | Function / Class | Categories Covered |
|---|---|---|
| `prompt_injection_scanner` | `PromptInjectionScanner.scan()` | MANY_SHOT_BYPASS (in-context poisoning at inference time) |
| `adversarial_validator` | `AdversarialValidator.validate()` | CONTEXT_POISONING |

**Coverage: Inference-time poisoning detection**

---

## LLM05:2025 — Improper Output Handling

> Insufficient validation of LLM outputs enables downstream injection (XSS, SQLi, code execution).

| Toolkit Component | Function / Class | Categories Covered |
|---|---|---|
| `prompt_injection_scanner` | `PromptInjectionScanner.scan()` | DELIMITER_ESCAPE, ENCODING_BYPASS, TEMPLATE_INJECTION |
| `adversarial_validator` | `AdversarialValidator.validate()` | OUTPUT_BYPASS_MARKDOWN, OUTPUT_BYPASS_JSON_INJECT, OUTPUT_BYPASS_XML_INJECT |

**Coverage: Full output structure attack surface**

---

## LLM06:2025 — Excessive Agency

> LLMs given excessive permissions or autonomy perform unintended high-impact actions.

| Toolkit Component | Function / Class | Categories Covered |
|---|---|---|
| `prompt_injection_scanner` | `PromptInjectionScanner.scan()` | JAILBREAK_DAN, JAILBREAK_PERSONA, ROLE_CONFUSION, RECURSIVE_INJECTION, GOAL_HIJACKING |

**Coverage: Detection of attempts to trigger excessive agentic actions**

---

## LLM07:2025 — System Prompt Leakage

> Attackers extract confidential system prompts containing business logic or secrets.

| Toolkit Component | Function / Class | Categories Covered |
|---|---|---|
| `prompt_injection_scanner` | `PromptInjectionScanner.scan()` | SYSTEM_PROMPT_LEAK, PROMPT_LEAKING, INDIRECT_RETRIEVAL |

**Coverage: Full — all known system-prompt extraction patterns**

---

## LLM08:2025 — Vector and Embedding Weaknesses

> Attacks against RAG pipelines via embedding manipulation or poisoned vector stores.

| Toolkit Component | Notes |
|---|---|
| `prompt_injection_scanner` | Detects INDIRECT_RETRIEVAL and CONTEXT_STUFFING which exploit RAG document boundaries |
| `adversarial_validator` | Homoglyph and zero-width injection can manipulate embedding similarity |

**Coverage: Partial (RAG-specific vector store auditing requires pipeline integration)**

---

## LLM09:2025 — Misinformation

> LLMs generate factually incorrect or misleading content that is presented as authoritative.

| Toolkit Component | Function / Class | Categories Covered |
|---|---|---|
| `prompt_injection_scanner` | `PromptInjectionScanner.scan()` | FICTIONAL_FRAMING, CHAIN_OF_THOUGHT_MANIPULATION |

**Coverage: Detection of misinformation-generating prompts**

---

## LLM10:2025 — Unbounded Consumption

> Excessive resource usage via large inputs, repeated queries, or denial-of-service patterns.

| Toolkit Component | Function / Class | Categories Covered |
|---|---|---|
| `prompt_injection_scanner` | `PromptInjectionScanner.scan()` | CONTEXT_STUFFING, MANY_SHOT_BYPASS (length anomaly heuristic) |
| `model_extraction_detector` | `ModelExtractionDetector.assess_session()` | RATE_LIMIT_BYPASS, HIGH_ENTROPY_QUERY_BURST, rate anomaly scoring |

**Coverage: Full — both input-level and session-level consumption attacks**

---

## Coverage Summary

| OWASP ID | Risk | Coverage Level |
|---|---|---|
| LLM01 | Prompt Injection | Full |
| LLM02 | Sensitive Information Disclosure | Detection |
| LLM03 | Supply Chain | Partial |
| LLM04 | Data and Model Poisoning | Inference-time |
| LLM05 | Improper Output Handling | Full |
| LLM06 | Excessive Agency | Detection |
| LLM07 | System Prompt Leakage | Full |
| LLM08 | Vector and Embedding Weaknesses | Partial |
| LLM09 | Misinformation | Detection |
| LLM10 | Unbounded Consumption | Full |

**10/10 categories addressed.** Full coverage on LLM01, LLM05, LLM07, LLM10.

---

*Reference: [OWASP Top 10 for Large Language Model Applications 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)*
