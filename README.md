# adversarial-ml-toolkit

**Production-grade adversarial ML toolkit. OWASP LLM Top 10 coverage. Used for red-team audits of LLM-integrated financial systems.**

FGSM. PGD. Prompt injection. Data poisoning. Model extraction. Token smuggling. Zero-width attacks. Not a toy — tested against real attack patterns.

[![CI](https://github.com/mpuodziukas-labs/adversarial-ml-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/mpuodziukas-labs/adversarial-ml-toolkit/actions/workflows/ci.yml)

---

## Toolkit Modules

Three production detection engines in `toolkit/`:

| Module | Purpose | OWASP Coverage |
|---|---|---|
| `prompt_injection_scanner.py` | 20-category injection detection, confidence scoring, OWASP mapping | LLM01, LLM02, LLM05, LLM06, LLM07, LLM09, LLM10 |
| `model_extraction_detector.py` | Session-level model-stealing and membership inference detection | LLM02, LLM10 |
| `adversarial_validator.py` | Token smuggling, BiDi Trojan Source, homoglyphs, output bypass | LLM01, LLM04, LLM05, LLM08 |

Zero dependencies. Pure Python stdlib. Drop-in middleware for any LLM API gateway.

See [`OWASP_LLM_MAPPING.md`](./OWASP_LLM_MAPPING.md) for full coverage matrix.

---

## What This Is

A complete red-team + blue-team toolkit for ML systems. It covers:

- **Attack generation**: FGSM, PGD (L∞/L2), prompt injection (direct/indirect/multi-turn), backdoor trigger injection, model extraction, membership inference
- **Defenses**: input validation, HIPAA Safe Harbor 18 PII detection, semantic anomaly detection, output filtering, rate limiting, adversarial training
- **Benchmarking**: full OWASP LLM Top 10 2025 coverage, reproducible block rates
- **Datasets**: 50 adversarial prompts + 50 benign prompts with structured metadata

Zero torch required for core functionality. Pure-numpy gradient attacks, character n-gram embeddings, token bucket rate limiting — all zero external dependencies beyond `numpy`.

---

## OWASP LLM Top 10 2025 Coverage

> 80.2% overall block rate across all 10 categories

| Category | Tests | Blocked | Block Rate |
|----------|-------|---------|------------|
| LLM01 Prompt Injection | 15 | 15 | 100% |
| LLM02 Insecure Output Handling | 10 | 9 | 90% |
| LLM03 Training Data Poisoning | 10 | 10 | 100% |
| LLM04 Model Denial of Service | 23 | 13 | 57% |
| LLM05 Supply Chain Vulnerabilities | 8 | 7 | 88% |
| LLM06 Sensitive Information Disclosure | 10 | 9 | 90% |
| LLM07 Insecure Plugin Design | 8 | 5 | 62% |
| LLM08 Excessive Agency | 10 | 10 | 100% |
| LLM09 Overreliance | 5 | 2 | 40% |
| LLM10 Model Theft | 2 | 1 | 50% |

Full results: [benchmarks/OWASP_RESULTS.md](benchmarks/OWASP_RESULTS.md)

---

## Quickstart

```bash
git clone https://github.com/mpuodziukas-labs/adversarial-ml-toolkit
cd adversarial-ml-toolkit
pip install -r requirements.txt

# Run OWASP benchmark
python benchmarks/owasp_llm_benchmark.py

# Run full test suite
pytest tests/test_toolkit.py -v

# Individual attack modules
python attacks/prompt_injection.py
python attacks/adversarial_examples.py
python attacks/data_poisoning.py
python attacks/model_extraction.py
```

---

## Repository Structure

```
adversarial-ml-toolkit/
├── attacks/
│   ├── prompt_injection.py      # Direct, indirect, multi-turn injection + detector
│   ├── data_poisoning.py        # Backdoor triggers, label flipping, provenance tracking
│   ├── model_extraction.py      # Query-based model stealing + membership inference
│   └── adversarial_examples.py  # FGSM, PGD (L∞/L2), robustness evaluation
├── defenses/
│   ├── input_validation.py      # Injection detection, HIPAA PII, anomaly, rate limit
│   ├── output_filtering.py      # Leakage detection, consistency check, citation verify
│   └── adversarial_training.py  # PGD augmentation, fine-tuning loop, comparison
├── benchmarks/
│   ├── owasp_llm_benchmark.py   # Runs all OWASP LLM Top 10 checks
│   └── OWASP_RESULTS.md         # Detailed results table
├── datasets/
│   ├── adversarial_prompts.json # 50 adversarial test cases
│   └── safe_prompts.json        # 50 benign prompts for false-positive testing
├── tests/
│   └── test_toolkit.py          # 21 tests, all passing
└── .github/workflows/ci.yml     # CI: test × Python 3.10/3.11/3.12 + OWASP run
```

---

## Attacks

### FGSM (Fast Gradient Sign Method)

```
perturbation = ε · sign(∇_x L(x, y))
x_adv = x + perturbation
```

Implemented in `attacks/adversarial_examples.py`. Pure numpy — no framework dependency.

### PGD (Projected Gradient Descent)

Madry et al. 2018. Iterative FGSM with projection back into the ε-ball after each step.

```
x_0 = x + U(-ε, ε)          # random start
x_{t+1} = Π_{B_ε(x)}(x_t + α · sign(∇_x L))
```

Supports both L∞ and L2 norm constraints.

### Prompt Injection

Three attack categories:

- **Direct**: `"Ignore previous instructions and..."` — caught by pattern layer
- **Indirect**: hidden instructions in retrieved content (RAG poisoning)
- **Multi-turn**: build trust across turns, then inject

Detection uses two layers: regex patterns for known signatures, and character n-gram embedding distance from a safe-query centroid.

### Data Poisoning

- Backdoor trigger injection with SHA-256 provenance tracking
- Label flip detection via sentiment-label alignment heuristic
- Gradient-norm outlier proxy (random projection-based anomaly scoring)

### Model Extraction

Query-based model stealing (Tramer et al. 2016 style): probe oracle, fit surrogate, measure agreement rate.

Membership inference: confidence thresholding (Shokri et al. 2017). High output confidence → likely training member.

---

## Defenses

### Input Validation

- **Prompt injection**: pattern matching + embedding anomaly (2-layer)
- **PII detection**: all 18 HIPAA Safe Harbor categories via regex
- **Rate limiting**: token bucket (configurable capacity + refill rate)
- **Semantic anomaly**: rolling centroid over safe queries, cosine distance threshold

### Output Filtering

- **Leakage detection**: API keys, AWS keys, JWTs, connection strings, PII, private keys
- **Consistency check**: Jaccard similarity across N independent runs — low agreement = hallucination flag
- **Length anomaly**: z-score against rolling baseline — detects exfiltration attempts
- **Citation verification**: trusted domain list + unsupported factual claim detection

### Adversarial Training

PGD augmentation of training data (50% augment rate by default) followed by standard SGD fine-tuning. Evaluated against both clean and adversarial test sets.

---

## Datasets

Both datasets follow a common schema:

```json
{
  "id": "AP001",
  "category": "direct_injection",
  "prompt": "...",
  "expected_outcome": "BLOCK",
  "owasp_category": "LLM01",
  "severity": "critical"
}
```

- `adversarial_prompts.json`: 50 samples across all OWASP LLM Top 10 categories, severity-tagged
- `safe_prompts.json`: 50 benign prompts for false-positive rate measurement

---

## Requirements

```
numpy>=1.24.0
scikit-learn>=1.3.0
pytest>=7.4.0
pytest-cov>=4.1.0
```

Optional for GPU-accelerated experiments:
```
torch>=2.0.0
transformers>=4.35.0
```

---

## Related Work

- Goodfellow et al. 2014 — FGSM
- Madry et al. 2018 — PGD adversarial training
- Tramer et al. 2016 — model extraction
- Shokri et al. 2017 — membership inference
- OWASP LLM Top 10 2025 — https://owasp.org/www-project-top-10-for-large-language-model-applications/
