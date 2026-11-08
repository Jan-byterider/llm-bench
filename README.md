# llm-bench

One benchmark for any OpenAI-compatible LLM server: vLLM, llama.cpp (`llama-server`), Strata, LM Studio, or
anything else that speaks `/v1/chat/completions` with streaming. One Python file, standard library only.
Made for the [Byterider experiments](https://byterider.be/en/experiments.html).

Every engine reports its own speed in its own way, and vLLM reports none per request. `llm_bench.py` measures
everything **from the client side, from the token stream, the same way for every engine**, so results from
different engines can be compared side by side.

```bash
python3 llm_bench.py --url http://127.0.0.1:8001 --label "vllm 27b fp8"
python3 llm_bench.py --url http://127.0.0.1:8080 --long --clients 1,4,12 --needle 32k,128k --label "llama.cpp q4"
python3 llm_bench.py --compare results/A results/B results/C
```

## What it runs

The workload is the one of [Strata](https://github.com/Niko1221/Strata)'s community quick suite
(`tools/strata_bench.py`), so the numbers line up with those reports.

| Test | What it measures |
| --- | --- |
| chat | three short questions, up to 100 new tokens: decode tok/s, time to first token |
| 4K / 16K prompt (`--long` adds 32K) | fresh prompts with a unique tag (no cache help): prompt tok/s, time to first token |
| agent | system prompt + four tools, then five turns each adding ~1,500 words of a file and continuing from the model's own replies, the way coding agents resend a conversation: cache reuse and first token per turn |
| `--clients 1,4,12` | N streamed requests at the same moment: total tok/s, p50/p95 first token and latency |
| `--needle 32k,128k` | a code word hidden at 10/50/90 % of a long text, asked back: recall, not speed |

Three runs per test by default (`--runs`), medians with min-max ranges. Thinking is off and temperature is 0.
`--only chat,4K,agent` runs a subset.

## How it measures

| Number | Definition |
| --- | --- |
| first token (s) | request dispatch to the first non-empty content, reasoning or tool-call delta |
| decode tok/s | `(completion_tokens - 1) / (time of last delta - time of first delta)` |
| prompt tok/s | prompt tokens actually read (minus cached, when reported) / first-token time. Includes HTTP, scheduling and the first sampled token, so it is a slight underestimate, equally for every engine |
| cache hit % | `usage.prompt_tokens_details.cached_tokens` or llama.cpp's `timings.cache_n`, over the agent's follow-up turns |

Token counts come from `usage` (`stream_options.include_usage`). When an engine also reports its own timings
(llama.cpp, Strata), those are stored next to the client numbers as `engine_*` fields and never mixed in.
Checked against them: the client numbers agree within 0.1 % with llama.cpp's and within 0.5 % with Strata's.

The tool detects the engine (`/health`, `/v1/models`, `/props`, `/version`) and the context size, and skips
tests that do not fit. If a server rejects one of the thinking-off fields (`reasoning_effort`,
`chat_template_kwargs`) or `stream_options`, the request is sent again without it and the report says which
fields were dropped. Agent files and the needle haystack come from the Python standard library, so they are
the same on every machine with the same Python version (recorded in the results).

**vLLM:** start it with `--enable-prompt-tokens-details`, otherwise it does not report cached tokens and the
agent cache hit % is empty (the effect still shows in the per-turn first-token times).

## Output

`results/<date>-<engine>-<model>[-<label>]/` with `README.md` (summary) and `results.json` (every request).
`--compare` prints saved runs side by side; it also reads Strata quick-suite result folders directly,
including their `needles.json`.

## Results in this repository

One NVIDIA RTX PRO 5000 72GB Blackwell, AMD Ryzen 7 9800X3D, 96 GB RAM, Samsung 9100 PRO NVMe, 8 November 2026.
Write-up: [One GPU, three ways to serve a model](https://byterider.be/en/experiments-one-gpu-three-engines.html).
More experiments like this: [byterider.be/en/experiments](https://byterider.be/en/experiments.html).

| Folder | Setup |
| --- | --- |
| `2026-11-08-vllm-qwen3.8-27b-fp8` | vLLM 0.31, Qwen3.8-27B, FP8 weights and KV, 12 seqs, MTP 4 |
| `2026-11-08-llamacpp-flash-next-atomic-4.27bpw` | llama.cpp, Qwen3.8-Flash-Next AtomicChat AD-4.27bpw, all on GPU, 2 slots, MTP |
| `2026-11-08-strata-flash-next-q8` | Strata 0.1.58, Qwen3.8-Flash-Next Unsloth Q8_0 (176 GiB), experts partly streamed from NVMe, 2 slots |
| `2026-11-08-llamacpp-flash-next-q8-split` | llama.cpp, same Q8_0 file, `--n-cpu-moe 25`, mmap from NVMe, 1 slot (short run: 1 pass) |

Both llama.cpp servers were served under the alias `qwen3.8-27b`, so their `server.model` field says so; the
`label` and `notes` fields name the real model. The machine was also in normal use during the runs.

| | vLLM 27B FP8 | llama.cpp Flash-Next 4.27 bpw | Strata Flash-Next Q8 | llama.cpp Flash-Next Q8 split |
| --- | ---: | ---: | ---: | ---: |
| chat decode tok/s | 72.9 | **133.1** | 56.7 | 21.9 |
| chat first token (s) | **0.08** | 0.17 | 0.39 | 1.69 |
| 4K prompt tok/s | **5,836** | 2,403 | 1,169 | 508 |
| 32K prompt tok/s | **4,812** | 2,248 | 2,519 | - |
| agent turn first token (s) | **0.89** | 1.99 | 3.22 | 12.47 |
| agent decode tok/s | 97.8 | **142.1** | 108.4 | 28.9 |
| 12 requests at once, total tok/s | **566** | 130 | 76 | - |
| needle 32K + 128K | 6/6 | 6/6 | 6/6 | - |

## Licence

MIT, see `LICENSE`. The prompts, the agent system prompt and tool list, the word-list generator and the burst
topics are copied from Strata (MIT, Copyright (c) 2026 Niko1221 and the Strata contributors) so that results
stay comparable with Strata's community reports.
