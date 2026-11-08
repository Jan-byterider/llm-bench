# llama.cpp - qwen3.8-27b - llama.cpp flash-next q8 split ncmoe25

2026-11-08T12:13:34, `llm_bench.py` (llm-bench-1), 1 runs per test, medians (min-max). Speeds are measured by the client from the token stream, the same way for every engine; engine-reported speeds are listed separately where the engine sends them.

- Server: http://127.0.0.1:8001 - engine **llama.cpp** b1-2a66703; context 65536 (/props n_ctx (per slot)); slots 1
- GPU: NVIDIA RTX PRO 5000 72GB Blackwell (71 GiB, driver 595.99.02); Python 3.12.3 (agent files and needle haystack come from its stdlib)
- Thinking off; temperature 0
- Notes: unsloth Flash-Next Q8_0 on llama.cpp (LM Studio backend 2.60.0), -ngl 99 --n-cpu-moe 25: experts of 25 of 48 layers in RAM (CPU compute), the rest on GPU; mmap pages the overflow from NVMe; -np 1 -c 65536, KV q8_0, MTP draft; GPU memory used after load 68742 MiB; 89 GiB RAM; short run (1 run, chat/4K/agent)

## Summary

| Metric | Median (range) |
| --- | --- |
| chat decode tok/s | 21.9 (19.2-26.1) |
| chat first token (s) | 1.69 (1.56-1.78) |
| 4K prompt tok/s | 507.8 (507.8-507.8) |
| 4K first token (s) | 7.92 (7.92-7.92) |
| agent cache hit % | 69.0 (69.0-69.0) |
| agent turn first token (s) | 12.47 (11.15-13.94) |
| agent last turn first token (s) | 12.33 (12.33-12.33) |
| agent decode tok/s | 28.9 (25.1-31.3) |

Engine-reported (not comparable across engines):

- chat engine decode tok/s: 21.9 (19.2-26.1)
- 4K engine prompt tok/s: 511.7 (511.7-511.7)

## Agent session, run 1

| Turn | Prompt tokens | Cached | First token (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 645 | 0 | 6.595 | 24.0 |
| 1 | 3466 | 655 | 12.49 | 25.1 |
| 2 | 6487 | 3529 | 12.474 | 27.6 |
| 3 | 8988 | 6550 | 11.149 | 31.3 |
| 4 | 12194 | 9051 | 13.94 | 28.9 |
| 5 | 15279 | 12257 | 12.333 | 29.9 |

Suite time 127 s. Every request: `results.json`.
