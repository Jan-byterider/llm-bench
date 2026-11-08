# vllm - qwen3.8-27b - vllm qwen3.8-27b-11-031-b8192

2026-11-08T10:55:23, `llm_bench.py` (llm-bench-1), 3 runs per test, medians (min-max). Speeds are measured by the client from the token stream, the same way for every engine; engine-reported speeds are listed separately where the engine sends them.

- Server: http://127.0.0.1:8001 - engine **vllm** 0.31.0; context 196608 (/v1/models max_model_len); slots ?
- GPU: NVIDIA RTX PRO 5000 72GB Blackwell (71 GiB, driver 595.99.02); Python 3.12.3 (agent files and needle haystack come from its stdlib)
- Thinking off; temperature 0
- Notes: profile qwen3.8-27b-11-031-b8192: fp8_per_tensor KV fp8, 12 seqs, ctx 196608, batched 8192, spec {"method":"mtp","num_speculative_tokens":4}, prompt_tokens_details off; GPU memory used at start 62272 MiB; embedder/reranker as loaded; opencode traffic not stopped

## Summary

| Metric | Median (range) |
| --- | --- |
| chat decode tok/s | 72.9 (67.3-119.4) |
| chat first token (s) | 0.08 (0.08-0.08) |
| 4K prompt tok/s | 5835.9 (5831.2-5848.7) |
| 4K first token (s) | 0.69 (0.69-0.69) |
| 16K prompt tok/s | 5516.1 (5499.9-5526.7) |
| 16K first token (s) | 2.91 (2.91-2.92) |
| 32K prompt tok/s | 4811.7 (4787.5-4831.7) |
| 32K first token (s) | 6.66 (6.63-6.69) |
| agent cache hit % | - |
| agent turn first token (s) | 0.89 (0.63-1.19) |
| agent last turn first token (s) | 1.18 (1.18-1.19) |
| agent decode tok/s | 97.8 (72.2-120.3) |
| 1 clients total tok/s | 60.0 (57.7-62.4) |
| 1 clients first token p95 (s) | 0.09 |
| 4 clients total tok/s | 234.5 (217.7-234.6) |
| 4 clients first token p95 (s) | 0.16 |
| 12 clients total tok/s | 565.8 (530.0-571.6) |
| 12 clients first token p95 (s) | 0.26 |
| needle found | 6/6 |

## Agent session, run 1

| Turn | Prompt tokens | Cached | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 645 | - | 0.175 | 116.7 |
| 1 | 3478 | - | 0.632 | 98.1 |
| 2 | 6499 | - | 0.886 | 72.2 |
| 3 | 9000 | - | 0.811 | 107.2 |
| 4 | 12203 | - | 1.139 | 91.8 |
| 5 | 15288 | - | 1.18 | 120.3 |

The server did not report cached tokens (vLLM: --enable-prompt-tokens-details); cache reuse shows only in the first-token times.

## 1 concurrent clients

Total 60.0 (57.7-62.4) tok/s; per request 62.1 (59.6-64.6) tok/s; first token p50 0.088 / p95 0.088 s; latency p50 2.134 / p95 2.211 s; errors 0

## 4 concurrent clients

Total 234.5 (217.7-234.6) tok/s; per request 68.0 (57.8-77.5) tok/s; first token p50 0.155 / p95 0.159 s; latency p50 2.026 / p95 2.259 s; errors 0

## 12 concurrent clients

Total 565.8 (530.0-571.6) tok/s; per request 55.6 (48.0-66.0) tok/s; first token p50 0.259 / p95 0.262 s; latency p50 2.539 / p95 2.836 s; errors 0

## Needle

| Length | Depth | Found | Prompt tokens | Seconds |
| --- | ---: | --- | ---: | ---: |
| 32k | 10% | yes | 32174 | 6.95 |
| 32k | 50% | yes | 32174 | 5.54 |
| 32k | 90% | yes | 32175 | 5.539 |
| 128k | 10% | yes | 128423 | 46.171 |
| 128k | 50% | yes | 128421 | 45.215 |
| 128k | 90% | yes | 128421 | 42.239 |

Suite time 246 s. Every request: `results.json`.
