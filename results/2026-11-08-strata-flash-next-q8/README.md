# strata - qwen3.8-flash-next-unsloth-q8_0 - strata qwen3.8-flash-next-strata-q8

2026-11-08T11:16:35, `llm_bench.py` (llm-bench-1), 3 runs per test, medians (min-max). Speeds are measured by the client from the token stream, the same way for every engine; engine-reported speeds are listed separately where the engine sends them.

- Server: http://127.0.0.1:8001 - engine **strata** ; context 262144 (/health max_context); slots ?
- GPU: NVIDIA RTX PRO 5000 72GB Blackwell (71 GiB, driver 595.99.02); Python 3.12.3 (agent files and needle haystack come from its stdlib)
- Thinking off; temperature 0
- Notes: profile qwen3.8-flash-next-strata-q8 (engine 0.1.58, --batch 2: requests beyond 2 queue); GPU memory used at start 72358 MiB; same suite as the vLLM 27B run

## Summary

| Metric | Median (range) |
| --- | --- |
| chat decode tok/s | 56.7 (45.1-63.8) |
| chat first token (s) | 0.39 (0.32-0.52) |
| 4K prompt tok/s | 1169.2 (1145.9-1188.3) |
| 4K first token (s) | 3.44 (3.38-3.51) |
| 16K prompt tok/s | 2347.9 (2339.2-2352.5) |
| 16K first token (s) | 6.84 (6.83-6.87) |
| 32K prompt tok/s | 2518.8 (2515.1-2521.4) |
| 32K first token (s) | 12.72 (12.71-12.74) |
| agent cache hit % | 68.7 (68.7-68.8) |
| agent turn first token (s) | 3.22 (3.09-3.32) |
| agent last turn first token (s) | 3.20 (3.18-3.31) |
| agent decode tok/s | 108.4 (43.3-177.8) |
| 1 clients total tok/s | 52.1 (34.2-61.6) |
| 1 clients first token p95 (s) | 0.75 |
| 4 clients total tok/s | 71.5 (66.9-79.5) |
| 4 clients first token p95 (s) | 4.58 |
| 12 clients total tok/s | 75.7 (73.3-75.9) |
| 12 clients first token p95 (s) | 17.62 |
| needle found | 6/6 |

Engine-reported (not comparable across engines):

- chat engine decode tok/s: 56.9 (45.2-64.0)
- 4K engine prompt tok/s: 1174.5 (1151.2-1193.6)
- 16K engine prompt tok/s: 2354.4 (2345.7-2359.9)
- 32K engine prompt tok/s: 2523.3 (2519.6-2526.0)

## Agent session, run 1

| Turn | Prompt tokens | Cached | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 2.34 | 39.8 |
| 1 | 3429 | 620 | 3.24 | 43.3 |
| 2 | 6450 | 3493 | 3.292 | 49.3 |
| 3 | 8951 | 6443 | 3.288 | 75.4 |
| 4 | 12157 | 9014 | 3.317 | 82.0 |
| 5 | 15231 | 12210 | 3.309 | 84.4 |

## 1 concurrent clients

Total 52.1 (34.2-61.6) tok/s; per request 68.9 (42.6-79.7) tok/s; first token p50 0.602 / p95 0.747 s; latency p50 2.455 / p95 3.615 s; errors 0

## 4 concurrent clients

Total 71.5 (66.9-79.5) tok/s; per request 40.9 (37.4-49.7) tok/s; first token p50 2.153 / p95 4.584 s; latency p50 5.337 / p95 7.643 s; errors 0

## 12 concurrent clients

Total 75.7 (73.3-75.9) tok/s; per request 42.9 (37.5-49.7) tok/s; first token p50 8.607 / p95 17.622 s; latency p50 11.659 / p95 20.459 s; errors 0

## Needle

| Length | Depth | Found | Prompt tokens | Seconds |
| --- | ---: | --- | ---: | ---: |
| 32k | 10% | yes | 32174 | 13.555 |
| 32k | 50% | yes | 32174 | 12.867 |
| 32k | 90% | yes | 32175 | 12.84 |
| 128k | 10% | yes | 128423 | 50.722 |
| 128k | 50% | yes | 128421 | 49.112 |
| 128k | 90% | yes | 128421 | 43.302 |

Suite time 441 s. Every request: `results.json`.
