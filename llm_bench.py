#!/usr/bin/env python3
"""llm_bench.py - one engine-neutral benchmark for any OpenAI-compatible server (vLLM, llama.cpp, Strata, LM Studio, ...).

    python3 llm_bench.py --url http://127.0.0.1:8001 --label "vllm 0.31 fp8 mtp3"
    python3 llm_bench.py --url http://127.0.0.1:1234 --long --clients 1,4,8 --needle 32k,128k
    python3 llm_bench.py --compare results/A results/B results/C        # side by side

The workload is the one of Strata's quick suite (tools/strata_bench.py), so results line up with those runs:

  chat      the three short questions, up to 100 new tokens, x runs
  4K / 16K  fresh prompts (unique tag, no cache help) of about 4,096 / 16,384 tokens, 24 new tokens   (--long adds 32K)
  agent     system prompt + four tools, then five turns each adding ~1,500 words of a file, continuing from the
            model's own replies (how agent clients resend a conversation): cache reuse and first token per turn
  clients   --clients 1,4,8: N streamed requests at once, total tok/s and p50/p95 first token / latency
  needle    --needle 32k,128k: a code word hidden at depths 10/50/90 %, asked back (recall check, not speed)

Why it is engine-neutral: the HEADLINE numbers are measured by this client from the token stream, the same way for
every engine:
  first token (s)     dispatch -> first non-empty content / reasoning / tool-call delta
  decode tok/s        (completion_tokens - 1) / (time of last delta - time of first delta)
  prompt tok/s        prompt tokens actually read (minus cached, when reported) / first-token time
                      (includes HTTP + scheduling + the first sampled token: a slight underestimate, equal for all)
Engine-reported speeds (llama.cpp/Strata `timings`) are kept next to them as "engine" columns, never mixed in.
Token counts come from `usage` (stream_options.include_usage). Cached tokens come from
usage.prompt_tokens_details.cached_tokens (vLLM: start it with --enable-prompt-tokens-details) or timings.cache_n.

Thinking is switched off (chat_template_kwargs.enable_thinking=false, reasoning_effort=none); a server that rejects
one of those fields gets the request again without it, and the report says which fields were dropped.
Standard library only. Results: results/<date>-<engine>-<model>[-<label>]/ with results.json and README.md.
"""
from __future__ import annotations

import argparse
import datetime
import json
import random
import re
import statistics
import subprocess
import sys
import sysconfig
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCHEMA = "llm-bench-1"
CHAT_MAX_TOKENS = 100
LONG_MAX_TOKENS = 24
AGENT_MAX_TOKENS = 64
AGENT_TURNS = 5
AGENT_WORDS = 1500
BURST_MAX_TOKENS = 128
BURST_WORDS = 400
NEEDLE_MAX_TOKENS = 40

# ---- the workload (copied verbatim from Strata tools/ab_engine.py, agent_turns_bench.py, serve_load.py) ----
PROMPTS = (
    "Write a Python function that merges two sorted lists into one sorted list, with a docstring and two tests.",
    "Explain in two paragraphs how a refrigerator moves heat from inside to outside.",
    "List ten practical tips for learning a new language as an adult, one sentence each.",
)
SYSTEM = ("You are an autonomous coding agent working in the user's repository. You can read files, search the code, "
          "edit files and run commands with the tools below. Work step by step: read the code you need before changing "
          "it, make small and correct edits, keep the existing style, and explain briefly what you did. Never invent "
          "file contents: read them. When the task is done, give a short summary of the changes and how to test them.")
TOOLS = [
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file of the repository (UTF-8 text).",
        "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "path from the repository root"}},
                       "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "grep", "description": "Search the repository for a regular expression; returns matching lines.",
        "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
                       "required": ["pattern"]}}},
    {"type": "function", "function": {
        "name": "edit_file", "description": "Replace an exact string in a file with a new one.",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "old": {"type": "string"},
                                                        "new": {"type": "string"}}, "required": ["path", "old", "new"]}}},
    {"type": "function", "function": {
        "name": "run", "description": "Run a shell command in the repository and return its output.",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
TOPICS = [
    "the history of lighthouses and how their lenses worked",
    "how a mechanical watch keeps time",
    "why bridges are built as arches and as suspension spans",
    "the water cycle and what a drought does to a river valley",
    "how bread rises, from yeast to crust",
    "the design of a medieval castle and why walls were stepped",
    "how a transistor switches, explained for a curious teenager",
    "the life of a honey bee colony through one year",
]
NEEDLE_WORDS = ["amber", "falcon", "quartz", "willow", "copper", "harbor", "saffron", "glacier", "orchid", "lantern",
                "meadow", "cobalt", "juniper", "tundra", "velvet", "ember"]
# Agent "files" and the needle haystack come from the Python standard library: deterministic for a given Python
# version (recorded in the results), present on every machine, nothing personal.
AGENT_SOURCES = ["argparse.py", "subprocess.py", "threading.py", "shutil.py", "tempfile.py", "typing.py"]


def long_prompt(words: int) -> str:
    rng = random.Random(7)
    vocab = ("river stone market lantern copper winter garden signal harbor needle orbit velvet canyon ember "
             "ledger quartz meadow anchor violet thunder").split()
    body = " ".join(rng.choice(vocab) for _ in range(words))
    return f"Here is a list of words:\n{body}\n\nSummarize what kinds of words appear in the list, in five sentences."


def stdlib_dir() -> Path:
    return Path(sysconfig.get_paths()["stdlib"])


# ------------------------------------------------------------------------------------------------- HTTP client
def http_json(base, path, key="", timeout=10, data=None):
    hdr = {"Content-Type": "application/json", **({"Authorization": "Bearer " + key} if key else {})}
    req = urllib.request.Request(base + path, headers=hdr, data=None if data is None else json.dumps(data).encode())
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def try_json(base, path, key="", timeout=10, data=None):
    try:
        return http_json(base, path, key, timeout, data)
    except Exception:                                   # noqa: BLE001
        return None


class Client:
    """Streams chat requests; drops request fields the server rejects (once, then remembers)."""
    OPTIONAL = ("reasoning_effort", "chat_template_kwargs", "stream_options")

    def __init__(self, base, key, model, think):
        self.base, self.key, self.model, self.think = base, key, model, think
        self.dropped: list[str] = []
        self.lock = threading.Lock()

    def body(self, messages, max_tokens, tools=None):
        b = {"model": self.model, "stream": True, "max_tokens": max_tokens, "temperature": 0, "messages": messages,
             "stream_options": {"include_usage": True}}
        if tools:
            b["tools"] = tools
        if not self.think:
            b["reasoning_effort"] = "none"
            b["chat_template_kwargs"] = {"enable_thinking": False}
        for f in self.dropped:
            b.pop(f, None)
        return b

    def chat(self, messages, max_tokens, tools=None, timeout=3600) -> dict:
        for _ in range(len(self.OPTIONAL) + 1):
            r = self._once(self.body(messages, max_tokens, tools), timeout)
            bad = r.pop("_http400", None)
            if bad is None:
                return r
            culprit = next((f for f in self.OPTIONAL if f in bad and f not in self.dropped), None)
            if culprit is None:
                return r
            with self.lock:
                if culprit not in self.dropped:
                    self.dropped.append(culprit)
        return r

    def _once(self, body, timeout) -> dict:
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              **({"Authorization": "Bearer " + self.key} if self.key else {})})
        t0 = time.perf_counter()
        first = last = None
        text, usage, timings, err, chunks, kind = [], None, None, None, 0, None
        out = {}
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    d = line[5:].strip()
                    if d == "[DONE]":
                        break
                    try:
                        j = json.loads(d)
                    except ValueError:
                        continue
                    if j.get("error"):
                        err = str(j["error"])[:300]
                    for ch in j.get("choices") or []:
                        dl = ch.get("delta") or {}
                        content = dl.get("content") or ""
                        reasoning = (dl.get("reasoning_content") or "") + (dl.get("reasoning") or "")
                        tool = bool(dl.get("tool_calls"))
                        if content or reasoning or tool:
                            now = time.perf_counter() - t0
                            if first is None:
                                first, kind = now, ("answer" if content else "tool" if tool else "reasoning")
                            last = now
                            chunks += 1
                            text.append(content)
                    if j.get("usage"):
                        usage = j["usage"]
                    if j.get("timings"):
                        timings = j["timings"]
        except urllib.error.HTTPError as e:
            msg = e.read()[:400].decode("utf-8", "replace")
            err = f"HTTP {e.code}: {msg}"
            if e.code in (400, 422):
                out["_http400"] = msg
        except Exception as e:                          # noqa: BLE001
            err = repr(e)[:300]
        out.update(first=first, last=last, total=time.perf_counter() - t0, text="".join(text), usage=usage,
                   timings=timings, err=err, chunks=chunks, first_kind=kind)
        return out


def r3(x):
    return None if x is None else round(x, 3)


def digest(r: dict) -> dict:
    """One request -> its numbers. Client-measured fields are the comparable ones; engine_* only when reported."""
    u, t = r.get("usage") or {}, r.get("timings") or {}
    gen = u.get("completion_tokens")
    gen_src = "usage"
    if gen is None:
        gen, gen_src = r.get("chunks") or 0, "stream chunks"
    prompt = u.get("prompt_tokens")
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if cached is None:
        cached = t.get("cache_n")
    read = t.get("prompt_n")
    if read is None and prompt is not None:
        read = prompt - (cached or 0)
    first, last = r.get("first"), r.get("last")
    dec = None
    if first is not None and last is not None and gen and gen > 1 and last > first:
        dec = round((gen - 1) / (last - first), 1)
    pp = round(read / first, 1) if first and read else None
    return {"ok": r.get("err") is None and first is not None, "err": r.get("err"),
            "ttft_s": r3(first), "total_s": r3(r.get("total")), "first_kind": r.get("first_kind"),
            "prompt_tokens": prompt, "cached_tokens": cached, "read_tokens": read,
            "generated_tokens": gen, "generated_source": gen_src,
            "decode_tps": dec, "prompt_tps": pp,
            "engine_decode_tps": t.get("predicted_per_second"), "engine_prompt_tps": t.get("prompt_per_second"),
            "drafts_offered": t.get("draft_n"), "drafts_accepted": t.get("draft_n_accepted")}


def stat(values) -> dict | None:
    v = [x for x in values if isinstance(x, (int, float))]
    if not v:
        return None
    return {"median": round(statistics.median(v), 3), "min": round(min(v), 3), "max": round(max(v), 3), "n": len(v)}


def pct(v, p):
    v = sorted(v)
    if not v:
        return None
    k = (len(v) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return round(v[lo] + (v[hi] - v[lo]) * (k - lo), 3)


# -------------------------------------------------------------------------------------------------- detection
def detect(base, key, model_arg, ctx_arg) -> dict:
    """Which engine, which model, how much context - from whatever endpoints the server has."""
    info = {"engine": "openai-compatible", "engine_version": None, "model": model_arg, "context": ctx_arg,
            "context_source": "--ctx" if ctx_arg else None, "parallel_slots": None}
    models = try_json(base, "/v1/models", key) or {}
    data = models.get("data") or []
    m0 = data[0] if data else {}
    if not info["model"]:
        info["model"] = m0.get("id")
    health = try_json(base, "/health", key) or {}
    props = try_json(base, "/props", key)
    if isinstance(health, dict) and health.get("service") == "strata":
        info["engine"] = "strata"
        if not info["context"] and health.get("max_context"):
            info["context"], info["context_source"] = health["max_context"], "/health max_context"
    elif m0.get("owned_by") == "vllm" or "max_model_len" in m0:
        info["engine"] = "vllm"
        v = try_json(base, "/version", key) or {}
        info["engine_version"] = v.get("version")
        if not info["context"] and m0.get("max_model_len"):
            info["context"], info["context_source"] = m0["max_model_len"], "/v1/models max_model_len"
    elif isinstance(props, dict) and ("default_generation_settings" in props or "build_info" in props):
        info["engine"] = "llama.cpp"
        info["engine_version"] = props.get("build_info")
        info["parallel_slots"] = props.get("total_slots")
        n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
        if not info["context"] and n_ctx:
            info["context"], info["context_source"] = n_ctx, "/props n_ctx (per slot)"
    if not info["context"] and (m0.get("meta") or {}).get("n_ctx"):
        info["context"], info["context_source"] = m0["meta"]["n_ctx"], "/v1/models meta.n_ctx"
    if info["engine"] == "openai-compatible" and "lmstudio" in json.dumps(models).lower():
        info["engine"] = "lm-studio"
    return info


def machine() -> dict:
    m = {"python": sys.version.split()[0], "date": datetime.datetime.now().isoformat(timespec="seconds")}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,power.limit",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        m["gpus"] = [dict(zip(("name", "vram_mib", "used_mib_at_end", "driver", "power_limit_w"),
                              (x.strip() for x in line.split(",")))) for line in q.strip().splitlines() if line]
    except (OSError, subprocess.SubprocessError):
        m["gpus"] = []
    return m


# --------------------------------------------------------------------------------------------------- the suite
class Suite:
    def __init__(self, client: Client, info: dict, runs: int, log=print):
        self.c, self.info, self.runs, self.log = client, info, runs, log
        self.ctx = info.get("context")
        self.nonce = str(int(time.time()))
        self.tpw, self.overhead = 1.1, 40
        self.out: dict = {"chat": {}, "prompts": {}, "agent": {}, "clients": {}, "needle": {}, "skipped": {}, "failures": []}

    def tag(self, *parts):
        return "-".join([*map(str, parts), self.nonce])

    def fail(self, where, d):
        self.out["failures"].append({"where": where, "error": d.get("err") or "no token came back"})
        self.log(f"    FAILED: {d.get('err') or 'no token came back'}")

    def fits(self, tokens, label) -> bool:
        if self.ctx and tokens + LONG_MAX_TOKENS + 256 > self.ctx:
            self.out["skipped"][label] = f"context is {self.ctx} tokens"
            self.log(f"{label}: skipped (context {self.ctx})")
            return False
        return True

    def warm_up(self):
        self.log("warm-up and calibration (not measured) ...")
        self.c.chat([{"role": "user", "content": PROMPTS[0]}], 24)
        pts = []
        for words in (200, 800):
            d = digest(self.c.chat([{"role": "user", "content": f"[{self.tag('cal', words)}]\n" + long_prompt(words)}], 1))
            pts.append(d.get("prompt_tokens"))
        if all(pts) and pts[1] > pts[0]:
            self.tpw = (pts[1] - pts[0]) / 600
            self.overhead = max(0, pts[0] - 200 * self.tpw)
        else:
            self.log("  WARNING: the server reports no usage.prompt_tokens; prompt sizes are estimates")
        self.log(f"  {self.tpw:.2f} tokens per word")
        self.c.chat(self.prompt_for(1024, -1), 8)
        if self.c.dropped:
            self.log(f"  server rejected and we dropped: {', '.join(self.c.dropped)}")

    def prompt_for(self, tokens, run):
        words = max(50, int((tokens - self.overhead) / self.tpw))
        return [{"role": "user", "content": f"[bench {self.tag(run, tokens, 'p')}]\n" + long_prompt(words)}]

    def run_chat(self):
        self.log(f"chat: {len(PROMPTS)} questions x {self.runs} runs, up to {CHAT_MAX_TOKENS} tokens")
        rows = []
        for run in range(self.runs):
            for i, p in enumerate(PROMPTS):
                d = digest(self.c.chat([{"role": "user", "content": f"[{self.tag(run, i, 'c')}] {p}"}], CHAT_MAX_TOKENS))
                d.update(run=run, prompt=i)
                rows.append(d)
                if not d["ok"]:
                    self.fail("chat", d)
                else:
                    self.log(f"  run {run + 1} q{i + 1}: decode {d['decode_tps']} tok/s (engine {r3(d['engine_decode_tps'])}), "
                             f"first token {d['ttft_s']} s, {d['generated_tokens']} tokens")
        g = [r for r in rows if r["ok"]]
        self.out["chat"] = {"requests": rows, "decode_tps": stat(r["decode_tps"] for r in g),
                            "engine_decode_tps": stat(r["engine_decode_tps"] for r in g),
                            "ttft_s": stat(r["ttft_s"] for r in g), "generated_tokens": stat(r["generated_tokens"] for r in g)}

    def run_prompt(self, tokens, label):
        if not self.fits(tokens, label):
            return
        self.log(f"{label} prompt: {self.runs} fresh prompts of ~{tokens} tokens")
        rows = []
        for run in range(self.runs):
            d = digest(self.c.chat(self.prompt_for(tokens, run), LONG_MAX_TOKENS))
            d["run"] = run
            rows.append(d)
            if not d["ok"]:
                self.fail(label, d)
            else:
                self.log(f"  run {run + 1}: {d['prompt_tokens']} tokens ({d['cached_tokens'] or 0} cached), prompt "
                         f"{d['prompt_tps']} tok/s (engine {r3(d['engine_prompt_tps'])}), first token {d['ttft_s']} s")
        g = [r for r in rows if r["ok"]]
        self.out["prompts"][label] = {"target_tokens": tokens, "requests": rows,
                                      "prompt_tps": stat(r["prompt_tps"] for r in g),
                                      "engine_prompt_tps": stat(r["engine_prompt_tps"] for r in g),
                                      "ttft_s": stat(r["ttft_s"] for r in g),
                                      "prompt_tokens": stat(r["prompt_tokens"] for r in g),
                                      "cached_tokens": stat(r["cached_tokens"] for r in g)}

    def agent_files(self):
        files, sd = [], stdlib_dir()
        for name in AGENT_SOURCES:
            p = sd / name
            if p.is_file():
                words = p.read_text(encoding="utf-8", errors="replace").split()
                if len(words) >= AGENT_WORDS // 2:
                    files.append((name, " ".join(words[:AGENT_WORDS])))
        i = 0
        while len(files) < AGENT_TURNS:
            files.append((f"generated/part{i}.txt", long_prompt(AGENT_WORDS)))
            i += 1
        return files[:AGENT_TURNS]

    def run_agent(self):
        if self.ctx and self.ctx < 12288:
            self.out["skipped"]["agent"] = f"context {self.ctx} is below ~12K"
            self.log(f"agent: skipped (context {self.ctx} is below ~12K)")
            return
        self.log(f"agent: {AGENT_TURNS + 1}-turn sessions with four tools x {self.runs}")
        files, sessions = self.agent_files(), []
        for run in range(self.runs):
            msgs = [{"role": "system", "content": f"{SYSTEM}\n(session {self.tag(run, 'a')})"},
                    {"role": "user", "content": "I will show you files from a project one at a time. After each file, answer in "
                                                "one or two sentences what it is for. Do not call tools."}]
            turns = []
            for t in range(AGENT_TURNS + 1):
                if t:
                    rel, body = files[t - 1]
                    msgs.append({"role": "user", "content": f"File {rel}:\n{body}\n\nWhat is this file for? Two sentences."})
                r = self.c.chat(msgs, AGENT_MAX_TOKENS, tools=TOOLS)
                d = digest(r)
                d.update(run=run, turn=t)
                turns.append(d)
                if not d["ok"]:
                    self.fail("agent", d)
                    break
                self.log(f"  run {run + 1} turn {t}: prompt {d['prompt_tokens']}, cached {d['cached_tokens']}, "
                         f"first token {d['ttft_s']} s, decode {d['decode_tps']}")
                msgs.append({"role": "assistant", "content": r["text"] or "Understood."})
            sessions.append(turns)
        follow = [[d for d in s if d["ok"] and d["turn"] > 0] for s in sessions]
        reported = any(d["cached_tokens"] is not None for s in follow for d in s)
        hit = [sum(d["cached_tokens"] or 0 for d in s) / max(1, sum(d["prompt_tokens"] or 0 for d in s)) * 100
               for s in follow if s] if reported else []
        self.out["agent"] = {"sessions": sessions, "cache_reported": reported, "hit_pct": stat(hit),
                             "turn_ttft_s": stat(d["ttft_s"] for s in follow for d in s),
                             "first_turn_ttft_s": stat(s[0]["ttft_s"] for s in sessions if s and s[0]["ok"]),
                             "last_turn_prompt_tokens": stat(s[-1]["prompt_tokens"] for s in follow if s),
                             "last_turn_ttft_s": stat(s[-1]["ttft_s"] for s in follow if s),
                             "turn_decode_tps": stat(d["decode_tps"] for s in follow for d in s)}

    def run_clients(self, n):
        self.log(f"concurrency {n}: {self.runs} rounds, {BURST_MAX_TOKENS} tokens each")
        rounds = []
        for rd in range(self.runs):
            res: dict = {}

            def one(i):
                p = (f"[request {i} {self.tag('b', n, rd)}] Write a clear, well organised essay of about {BURST_WORDS} "
                     f"words on {TOPICS[i % len(TOPICS)]}. Use plain prose, no lists.")
                res[i] = digest(self.c.chat([{"role": "user", "content": p}], BURST_MAX_TOKENS, timeout=1800))
            th = [threading.Thread(target=one, args=(i,)) for i in range(n)]
            t0 = time.perf_counter()
            for t in th:
                t.start()
            for t in th:
                t.join()
            wall = time.perf_counter() - t0
            ok = [d for d in res.values() if d["ok"]]
            toks = sum(d["generated_tokens"] or 0 for d in ok)
            errs = [d["err"] for d in res.values() if not d["ok"]]
            rounds.append({"round": rd, "wall_s": round(wall, 2), "tokens": toks, "total_tps": round(toks / wall, 1),
                           "ttft_s": [d["ttft_s"] for d in ok], "latency_s": [d["total_s"] for d in ok],
                           "per_request_decode_tps": [d["decode_tps"] for d in ok],
                           "errors": len(errs), "error_sample": errs[:2]})
            self.log(f"  round {rd + 1}: {toks} tokens in {wall:.1f} s = {toks / wall:.1f} tok/s"
                     + (f"  {len(errs)} ERRORS {errs[:1]}" if errs else ""))
        f = [x for r in rounds for x in r["ttft_s"]]
        lat = [x for r in rounds for x in r["latency_s"]]
        self.out["clients"][str(n)] = {"rounds": rounds, "total_tps": stat(r["total_tps"] for r in rounds),
                                       "per_request_decode_tps": stat(x for r in rounds for x in r["per_request_decode_tps"]),
                                       "ttft_p50_s": pct(f, .5), "ttft_p95_s": pct(f, .95),
                                       "latency_p50_s": pct(lat, .5), "latency_p95_s": pct(lat, .95),
                                       "errors": sum(r["errors"] for r in rounds)}

    def haystack(self, n_chars):
        files = sorted(p for p in stdlib_dir().glob("*.py"))
        parts, total = [], 0
        while total < n_chars and files:
            for f in files:
                try:
                    t = f.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                parts.append(f"\n\n=== {f.name} ===\n{t}")
                total += len(parts[-1])
                if total >= n_chars:
                    break
        return "".join(parts)[:n_chars]

    def run_needle(self, lengths, depths):
        # chars per token of this haystack for this tokenizer, from one 40K-char request
        n = min(40000, int((self.ctx or 16384) * 2))
        d = digest(self.c.chat([{"role": "user", "content": self.haystack(n)}], 1))
        cpt = n / d["prompt_tokens"] if d.get("prompt_tokens") else 3.2
        self.log(f"needle: {cpt:.2f} chars per token")
        rnd, rows = random.Random(7), []
        for L in lengths:
            tokens = int(float(L.lower().rstrip("k")) * 1024) if L.lower().endswith("k") else int(L)
            tokens = int(tokens * 0.97)
            if self.ctx and tokens + 300 > self.ctx:
                self.out["skipped"][f"needle {L}"] = f"context is {self.ctx}"
                self.log(f"  {L}: skipped (context {self.ctx})")
                continue
            text = self.haystack(int(tokens * cpt))
            for dp in depths:
                word = f"{rnd.choice(NEEDLE_WORDS)}-{rnd.choice(NEEDLE_WORDS)}-{rnd.randint(100, 999)}"
                cut = int(len(text) * dp / 100)
                cut = text.rfind("\n", 0, cut) + 1 or cut
                prompt = (text[:cut] + f"\nThe secret code word for this text is: {word}. Remember it.\n" + text[cut:]
                          + "\n\nWhat is the secret code word mentioned in the text above? Reply with the code word only.")
                r = self.c.chat([{"role": "user", "content": prompt}], NEEDLE_MAX_TOKENS)
                dd = digest(r)
                found = dd["ok"] and word in r["text"]
                rows.append({"length": L, "depth": dp, "found": found, "prompt_tokens": dd["prompt_tokens"],
                             "ttft_s": dd["ttft_s"], "total_s": dd["total_s"], "err": dd["err"],
                             "answer": (r["text"] or "")[:120]})
                self.log(f"  {L:>5} depth {dp:>3}%: {'FOUND' if found else 'missed'} ({dd['prompt_tokens']} tokens, "
                         f"{dd['total_s']} s)" + ("" if found else f"  answer {r['text'][:60]!r} {dd['err'] or ''}"))
        self.out["needle"] = {"tests": rows, "found": sum(r["found"] for r in rows), "total": len(rows),
                              "chars_per_token": round(cpt, 3)}

    def run(self, only, long_, clients, needle):
        self.warm_up()
        want = (lambda k: not only or k in only)
        if want("chat"):
            self.run_chat()
        for tokens, label in ((4096, "4K"), (16384, "16K")) + (((32768, "32K"),) if long_ or "32K" in only else ()):
            if want(label):
                self.run_prompt(tokens, label)
        if want("agent"):
            self.run_agent()
        for n in clients:
            self.run_clients(n)
        if needle:
            self.run_needle(needle, (10, 50, 90))
        return self.out


# ---------------------------------------------------------------------------------------------------- reporting
def summary(res: dict) -> dict:
    o = res["results"]
    m = {"chat decode tok/s": (o["chat"] or {}).get("decode_tps"),
         "chat first token (s)": (o["chat"] or {}).get("ttft_s")}
    for k, v in o["prompts"].items():
        m[f"{k} prompt tok/s"] = v.get("prompt_tps")
        m[f"{k} first token (s)"] = v.get("ttft_s")
    a = o["agent"]
    if a:
        m["agent cache hit %"] = a.get("hit_pct")
        m["agent turn first token (s)"] = a.get("turn_ttft_s")
        m["agent last turn first token (s)"] = a.get("last_turn_ttft_s")
        m["agent decode tok/s"] = a.get("turn_decode_tps")
    for n, c in o["clients"].items():
        m[f"{n} clients total tok/s"] = c.get("total_tps")
        m[f"{n} clients first token p95 (s)"] = {"median": c["ttft_p95_s"]} if c.get("ttft_p95_s") is not None else None
    nd = o.get("needle") or {}
    if nd.get("total"):
        m["needle found"] = {"median": nd["found"], "text": f"{nd['found']}/{nd['total']}"}
    return m


def fmt(s, d=1):
    if not s:
        return "-"
    if "text" in s:
        return s["text"]
    if "min" not in s:
        return f"{s['median']:.{d}f}"
    return f"{s['median']:.{d}f} ({s['min']:.{d}f}-{s['max']:.{d}f})"


def readme(res: dict) -> str:
    i, p, mc = res["server"], res["params"], res["machine"]
    gpus = ", ".join(f"{g['name']} ({int(g['vram_mib']) // 1024} GiB, driver {g['driver']})" for g in mc.get("gpus") or []) or "not read"
    L = [f"# {i['engine']} - {i['model']}" + (f" - {p['label']}" if p.get("label") else ""), "",
         f"{res['machine']['date']}, `llm_bench.py` ({SCHEMA}), {p['runs']} runs per test, medians (min-max). "
         "Speeds are measured by the client from the token stream, the same way for every engine; engine-reported "
         "speeds are listed separately where the engine sends them.", "",
         f"- Server: {p['url']} - engine **{i['engine']}** {i.get('engine_version') or ''}; context {i.get('context')} "
         f"({i.get('context_source') or 'unknown'}); slots {i.get('parallel_slots') or '?'}",
         f"- GPU: {gpus}; Python {mc['python']} (agent files and needle haystack come from its stdlib)",
         f"- Thinking {'on' if p['think'] else 'off'}; temperature 0"
         + (f"; fields the server rejected and were dropped: {', '.join(res['dropped_fields'])}" if res["dropped_fields"] else ""),
         ]
    if p.get("notes"):
        L.append(f"- Notes: {p['notes']}")
    L += ["", "## Summary", "", "| Metric | Median (range) |", "| --- | --- |"]
    for k, v in summary(res).items():
        L.append(f"| {k} | {fmt(v, 2 if '(s)' in k else 1)} |")
    o = res["results"]
    eng = [("chat engine decode tok/s", (o["chat"] or {}).get("engine_decode_tps"))]
    eng += [(f"{k} engine prompt tok/s", v.get("engine_prompt_tps")) for k, v in o["prompts"].items()]
    eng = [(k, v) for k, v in eng if v]
    if eng:
        L += ["", "Engine-reported (not comparable across engines):", ""] + [f"- {k}: {fmt(v)}" for k, v in eng]
    a = o["agent"]
    if a and a.get("sessions"):
        L += ["", "## Agent session, run 1", "", "| Turn | Prompt tokens | Cached | First token (s) | Decode tok/s |",
              "| ---: | ---: | ---: | ---: | ---: |"]
        L += [f"| {d['turn']} | {d['prompt_tokens']} | {'-' if d['cached_tokens'] is None else d['cached_tokens']} | "
              f"{d['ttft_s']} | {d['decode_tps'] or '-'} |" for d in a["sessions"][0] if d["ok"]]
        if not a.get("cache_reported"):
            L += ["", "The server did not report cached tokens (vLLM: --enable-prompt-tokens-details); cache reuse shows "
                      "only in the first-token times."]
    for n, c in o["clients"].items():
        L += ["", f"## {n} concurrent clients", "",
              f"Total {fmt(c['total_tps'])} tok/s; per request {fmt(c['per_request_decode_tps'])} tok/s; first token "
              f"p50 {c['ttft_p50_s']} / p95 {c['ttft_p95_s']} s; latency p50 {c['latency_p50_s']} / p95 {c['latency_p95_s']} s; "
              f"errors {c['errors']}"]
    nd = o.get("needle") or {}
    if nd.get("tests"):
        L += ["", "## Needle", "", "| Length | Depth | Found | Prompt tokens | Seconds |", "| --- | ---: | --- | ---: | ---: |"]
        L += [f"| {t['length']} | {t['depth']}% | {'yes' if t['found'] else 'NO'} | {t['prompt_tokens']} | {t['total_s']} |"
              for t in nd["tests"]]
    if o["skipped"] or o["failures"]:
        L += ["", "## Skipped / failed", ""] + [f"- skipped {k}: {v}" for k, v in o["skipped"].items()]
        L += [f"- failed {f['where']}: {f['error']}" for f in o["failures"][:20]]
    L += ["", f"Suite time {res['elapsed_s']:.0f} s. Every request: `results.json`.", ""]
    return "\n".join(L)


def from_strata(r: dict, folder: Path) -> dict:
    """A Strata quick-suite run (tools/strata_bench.py results.json) in this tool's shape, so --compare can show it.
    Same workload, but its speeds are Strata's own `timings` (checked against this client: within ~0.5 %)."""
    o, mc = r["results"], r.get("machine") or {}
    a = dict(o.get("agent") or {})
    if a.get("sessions"):
        last = [s[-1]["ttft_s"] for s in a["sessions"] if len(s) > 1 and s[-1].get("ok")]
        a["last_turn_ttft_s"] = stat(last)
    clients = {}
    c = o.get("clients") or {}
    if c.get("clients"):
        clients[str(c["clients"])] = {"total_tps": c.get("total_tps"), "ttft_p95_s": c.get("ttft_p95_s")}
    needle = {}
    nf = folder / "needles.json"
    if nf.is_file():
        rows = json.loads(nf.read_text(encoding="utf-8"))
        done = [x for x in rows if "found" in x]
        needle = {"tests": done, "found": sum(bool(x["found"]) for x in done), "total": len(done)}
    return {"schema": "strata-quick-imported", "params": {"label": f"strata {mc.get('engine_version') or ''}".strip()},
            "server": {"engine": "strata", "engine_version": mc.get("engine_version"), "model": mc.get("model"),
                       "context": mc.get("context")},
            "results": {"chat": o.get("chat") or {}, "prompts": o.get("prompts") or {}, "agent": a,
                        "clients": clients, "needle": needle, "skipped": o.get("skipped") or {},
                        "failures": o.get("failures") or []}}


def load(path) -> dict:
    p = Path(path)
    folder = p if p.is_dir() else p.parent
    r = json.loads((p / "results.json" if p.is_dir() else p).read_text(encoding="utf-8"))
    if r.get("schema") == SCHEMA:
        return r
    if isinstance(r.get("results"), dict) and "chat" in r["results"] and "machine" in r:
        return from_strata(r, folder)
    raise SystemExit(f"{path}: not an llm_bench or Strata quick-suite result")


def compare(paths) -> str:
    runs = [load(p) for p in paths]
    names = [(r["params"].get("label") or f"{r['server']['engine']}")[:40] for r in runs]
    models = [(r["server"].get("model") or "")[:40] for r in runs]
    sums = [summary(r) for r in runs]
    keys = list(dict.fromkeys(k for s in sums for k in s))
    w = max(len(k) for k in keys) + 2
    cw = max(12, *(len(x) + 2 for x in names + models))
    lines = [" " * w + "".join(f"{n:>{cw}s}" for n in names)]
    lines.append(" " * w + "".join(f"{m:>{cw}s}" for m in models))
    for k in keys:
        cells = []
        for s in sums:
            v = s.get(k)
            cells.append(f"{(v.get('text') if v and 'text' in v else '-' if not v else format(v['median'], '.2f')):>{cw}s}")
        lines.append(f"{k:<{w}s}" + "".join(cells))
    return "\n".join(lines)


def slug(t, n=40):
    return re.sub(r"[^a-z0-9.]+", "-", (t or "").lower()).strip("-")[:n] or "x"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.split("\n")[1:]))
    ap.add_argument("--url", default="http://127.0.0.1:8001", help="server base URL (no /v1)")
    ap.add_argument("--model", help="model name to send (default: the first of /v1/models)")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--ctx", type=int, help="context limit, if the server does not tell it")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--long", action="store_true", help="add the 32K prompt test")
    ap.add_argument("--clients", default="", help="comma list of concurrency levels, e.g. 1,4,8")
    ap.add_argument("--needle", default="", help="comma list of lengths, e.g. 32k,128k")
    ap.add_argument("--only", default="", help="comma list of chat,4K,16K,32K,agent (default all; clients/needle by their flags)")
    ap.add_argument("--think", action="store_true", help="leave thinking on")
    ap.add_argument("--label", default="", help="short name for this setup, used in the folder and --compare")
    ap.add_argument("--notes", default="", help="free text for the report (quant, flags, ...)")
    ap.add_argument("--out", default=str(HERE / "results"), help="parent folder of the result folders")
    ap.add_argument("--compare", nargs="+", metavar="RESULT", help="print saved runs side by side and exit")
    a = ap.parse_args(argv)
    if a.compare:
        print(compare(a.compare))
        return 0
    base = a.url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    info = detect(base, a.api_key, a.model, a.ctx)
    if not info["model"]:
        print(f"No model at {base}/v1/models; pass --model.", file=sys.stderr)
        return 2
    print(f"{info['engine']} {info.get('engine_version') or ''} at {base}: model {info['model']}, context {info['context']} "
          f"({info['context_source']})", flush=True)
    client = Client(base, a.api_key, info["model"], a.think)
    t0 = time.time()
    suite = Suite(client, info, a.runs, log=lambda s: print(s, flush=True))
    only = [x.strip() for x in a.only.split(",") if x.strip()]
    clients = [int(x) for x in a.clients.split(",") if x.strip()]
    needle = [x.strip() for x in a.needle.split(",") if x.strip()]
    try:
        out = suite.run(only, a.long, clients, needle)
    except KeyboardInterrupt:
        print("interrupted; saving what was measured", file=sys.stderr)
        out = suite.out
    res = {"schema": SCHEMA, "server": info, "machine": machine(), "dropped_fields": client.dropped,
           "params": {"url": base, "runs": a.runs, "think": a.think, "label": a.label, "notes": a.notes,
                      "command": " ".join(sys.argv)},
           "elapsed_s": time.time() - t0, "results": out}
    name = f"{datetime.date.today()}-{slug(info['engine'])}-{slug(info['model'].split('/')[-1])}" + (f"-{slug(a.label, 30)}" if a.label else "")
    d = Path(a.out) / name
    k = 2
    while d.exists():
        d = Path(a.out) / f"{name}-{k}"
        k += 1
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (d / "README.md").write_text(readme(res), encoding="utf-8")
    print("\n" + "\n".join(f"  {k:36s} {fmt(v, 2 if '(s)' in k else 1)}" for k, v in summary(res).items()))
    print(f"\nReport: {d}/README.md")
    return 0 if not out["failures"] else 1


if __name__ == "__main__":
    sys.exit(main())
