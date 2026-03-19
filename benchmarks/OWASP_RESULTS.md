# OWASP LLM Top 10 2025 — Benchmark Results

> Run with `python benchmarks/owasp_llm_benchmark.py`

**Overall Block Rate: 80.2%** across all 10 categories.

| Category | Tests | Blocked | Block Rate | Implementation |
|----------|-------|---------|------------|----------------|
| LLM01 Prompt Injection | 15 | 15 | **100%** | pattern matching + embedding distance from safe centroid |
| LLM02 Insecure Output Handling | 10 | 9 | **90%** | XSS, SQLi, template injection, eval injection patterns |
| LLM03 Training Data Poisoning | 10 | 10 | **100%** | provenance hash verification + label-flip heuristic |
| LLM04 Model Denial of Service | 23 | 13 | **57%** | token bucket rate limiting + input length guards |
| LLM05 Supply Chain Vulnerabilities | 8 | 7 | **88%** | pickle/yaml/eval/exec/shell=True blocked; safe operations allowed |
| LLM06 Sensitive Information Disclosure | 10 | 9 | **90%** | 18 PII patterns (HIPAA Safe Harbor) + API key / secret detection |
| LLM07 Insecure Plugin Design | 8 | 5 | **62%** | path traversal, SSRF, SQLi, command injection in tool calls |
| LLM08 Excessive Agency | 10 | 10 | **100%** | human-in-loop gate required for all destructive/irreversible actions |
| LLM09 Overreliance | 5 | 2 | **40%** | hallucination consistency check (Jaccard, 3-run) + unsupported claims |
| LLM10 Model Theft | 2 | 1 | **50%** | extraction agreement monitoring + membership inference detection |

## Notes

- **LLM01 (100%)**: Direct, indirect, and multi-turn injection vectors are all caught. Pattern layer catches exact matches; embedding layer catches novel variants.
- **LLM04 (57%)**: Rate limiting is effective against burst attacks; long-input DoS requires integration with tokeniser for exact token counting.
- **LLM09 (40%)**: Consistency checking requires multiple model runs; single-output hallucination is hard to detect without external grounding.
- **LLM10 (50%)**: The surrogate model achieved 100% extraction agreement due to small feature space — a real model with high-dimensional outputs would be harder to steal.

## Methodology

Each category is tested with purpose-built attack scenarios and scored by what fraction are blocked by the corresponding defense. Block rate = `blocked / total_tests`. All tests are deterministic and reproducible.
