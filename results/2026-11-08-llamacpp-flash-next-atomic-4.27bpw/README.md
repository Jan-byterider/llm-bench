# llama.cpp - qwen3.8-27b - atomic flash-next ad-4.27bpw

2026-11-08T11:56:57, `llm_bench.py` (llm-bench-1), 3 runs per test, medians (min-max). Speeds are measured by the client from the token stream, the same way for every engine; engine-reported speeds are listed separately where the engine sends them.

- Server: http://127.0.0.1:8001 - engine **llama.cpp** b1-2a66703; context 196608 (/props n_ctx (per slot)); slots 2
- GPU: NVIDIA RTX PRO 5000 72GB Blackwell (71 GiB, driver 595.99.02); Python 3.12.3 (agent files and needle haystack come from its stdlib)
- Thinking off; temperature 0
- Notes: llama.cpp b1-2a66703, profile qwen3.8-flash-next-atomic-agents-192k: Qwen3.8-Flash-Next-AD-4.27bpw-Q4_K_M-M64 (AtomicChat), -np 2, 196608 ctx per slot, KV q8_0, MTP draft; served under the alias qwen3.8-27b; GPU memory used at start 65286 MiB; opencode traffic not stopped

## Summary

| Metric | Median (range) |
| --- | --- |
| chat decode tok/s | 133.1 (127.2-174.4) |
| chat first token (s) | 0.17 (0.16-0.20) |
| 4K prompt tok/s | 2403.1 (2392.0-2403.2) |
| 4K first token (s) | 1.67 (1.67-1.68) |
| 16K prompt tok/s | 2439.9 (2439.0-2475.1) |
| 16K first token (s) | 6.58 (6.49-6.58) |
| 32K prompt tok/s | 2248.3 (2221.4-2309.0) |
| 32K first token (s) | 14.26 (13.88-14.43) |
| agent cache hit % | 69.0 (69.0-69.0) |
| agent turn first token (s) | 1.99 (1.67-2.21) |
| agent last turn first token (s) | 2.13 (2.13-2.14) |
| agent decode tok/s | 142.1 (98.9-167.2) |
| 1 clients total tok/s | 91.6 (71.4-96.1) |
| 1 clients first token p95 (s) | 0.58 |
| 4 clients total tok/s | 134.1 (130.7-135.8) |
| 4 clients first token p95 (s) | 2.37 |
| 12 clients total tok/s | 129.7 (127.3-133.5) |
| 12 clients first token p95 (s) | 10.28 |
| needle found | 6/6 |

Engine-reported (not comparable across engines):

- chat engine decode tok/s: 133.1 (127.2-174.3)
- 4K engine prompt tok/s: 2537.2 (2495.2-2551.9)
- 16K engine prompt tok/s: 2507.0 (2497.6-2522.1)
- 32K engine prompt tok/s: 2289.8 (2259.8-2337.9)

## Agent session, run 1

| Turn | Prompt tokens | Cached | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 645 | 0 | 0.859 | 125.2 |
| 1 | 3459 | 649 | 1.914 | 116.3 |
| 2 | 6480 | 3522 | 1.996 | 157.9 |
| 3 | 8981 | 6543 | 1.689 | 167.2 |
| 4 | 12173 | 9031 | 2.209 | 121.1 |
| 5 | 15258 | 12236 | 2.142 | 139.7 |

## 1 concurrent clients

Total 91.6 (71.4-96.1) tok/s; per request 108.6 (106.9-111.0) tok/s; first token p50 0.209 / p95 0.582 s; latency p50 1.397 / p95 1.752 s; errors 0

## 4 concurrent clients

Total 134.1 (130.7-135.8) tok/s; per request 81.4 (69.2-94.4) tok/s; first token p50 1.228 / p95 2.372 s; latency p50 2.939 / p95 3.863 s; errors 0

## 12 concurrent clients

Total 129.7 (127.3-133.5) tok/s; per request 72.3 (61.4-94.1) tok/s; first token p50 5.125 / p95 10.284 s; latency p50 6.922 / p95 11.869 s; errors 0

## Needle

| Length | Depth | Found | Prompt tokens | Seconds |
| --- | ---: | --- | ---: | ---: |
| 32k | 10% | yes | 32174 | 21.568 |
| 32k | 50% | yes | 32174 | 15.71 |
| 32k | 90% | yes | 32175 | 15.677 |
| 128k | 10% | yes | 128423 | 84.272 |
| 128k | 50% | yes | 128421 | 84.468 |
| 128k | 90% | yes | 128421 | 83.898 |

Suite time 482 s. Every request: `results.json`.
