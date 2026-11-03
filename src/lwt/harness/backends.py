"""Scientist backends: one multi-turn conversation per object. No tools are given to the model; it acts by writing
fenced blocks in plain text, which the harness parses (as upstream did with Codex's tools switched off).

Each `send` returns a Reply with the visible text, the reasoning summary, usage and the turn's cost in USD.
"""
import time
from dataclasses import dataclass, field

# USD per million tokens: input, output, cache read, 1-hour cache write (2x input).
PRICES = {
    "claude-fable-5-1": dict(input=10.0, output=50.0, cache_read=0.25, cache_write_1h=20.0),
    "claude-opus-5-5": dict(input=4.0, output=20.0, cache_read=0.20, cache_write_1h=8.0),
}

# OpenAI: USD per million tokens (third-party listings, Nov 2026; check openai.com/api/pricing). A request whose input
# exceeds long_context_over tokens bills at the long-context rates. OpenAI's input_tokens include cached tokens.
OPENAI_PRICES = {
    "gpt-6-astra": dict(input=10.0, cached=1.0, output=50.0, long_context_over=272_000,
                        long=dict(input=20.0, cached=2.0, output=75.0)),
}

INSTRUCTIONS = "Answer the user's request."   # upstream harness/instructions.md, verbatim


@dataclass
class Reply:
    text: str
    reasoning: str
    stop_reason: str
    model: str
    usage: dict
    usd: float
    seconds: float
    raw: dict = field(default_factory=dict)


class AnthropicBackend:
    """Claude via the Messages API. The conversation is append-only: every response's content blocks (thinking
    included) go back unchanged, which keeps the prompt cache and preserved thinking valid."""

    def __init__(self, model: str, effort: str = "xhigh", max_tokens: int = 64000, fallbacks: bool = False,
                 cache_ttl: str = "1h", system: str = INSTRUCTIONS):
        import anthropic
        if model not in PRICES:
            raise ValueError(f"no price table for {model}; add it to PRICES in backends.py")
        self.client = anthropic.Anthropic(max_retries=8)
        self.model, self.effort, self.max_tokens, self.fallbacks = model, effort, max_tokens, fallbacks
        self.cache_ttl, self.system = cache_ttl, system
        self.messages: list[dict] = []

    def send(self, user_text: str) -> Reply:
        self.messages.append({"role": "user", "content": user_text})
        kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=self.system, messages=self.messages,
                      thinking={"type": "adaptive", "display": "summarized"},
                      output_config={"effort": self.effort},
                      cache_control={"type": "ephemeral", "ttl": self.cache_ttl})
        t0 = time.time()
        if self.fallbacks:
            # On a safety decline the API re-runs the turn on Anthropic's recommended fallback model. The serving
            # model is recorded per turn, so analyses can drop or flag rescued turns.
            with self.client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                                  **kwargs) as stream:
                msg = stream.get_final_message()
        else:
            with self.client.messages.stream(**kwargs) as stream:
                msg = stream.get_final_message()
        self.messages.append({"role": "assistant", "content": msg.content})
        text = "".join(b.text for b in msg.content if b.type == "text")
        reasoning = "\n\n".join(b.thinking for b in msg.content if b.type == "thinking" and b.thinking)
        u = msg.usage
        usage = dict(input=u.input_tokens, output=u.output_tokens,
                     cache_read=u.cache_read_input_tokens or 0, cache_write=u.cache_creation_input_tokens or 0)
        p = PRICES[self.model]
        usd = (usage["input"] * p["input"] + usage["output"] * p["output"] + usage["cache_read"] * p["cache_read"]
               + usage["cache_write"] * p["cache_write_1h"]) / 1e6
        served = [getattr(it, "model", None) for it in (getattr(u, "iterations", None) or [])]
        return Reply(text=text, reasoning=reasoning, stop_reason=msg.stop_reason, model=msg.model, usage=usage,
                     usd=usd, seconds=round(time.time() - t0, 1),
                     raw={"id": msg.id, "served_models": served,
                          "stop_details": msg.stop_details.model_dump() if getattr(msg, "stop_details", None) else None})


def openai_usd(model: str, input_tokens: int, cached: int, output: int) -> float:
    p = OPENAI_PRICES[model]
    rates = p["long"] if input_tokens > p["long_context_over"] else p
    return ((input_tokens - cached) * rates["input"] + cached * rates["cached"] + output * rates["output"]) / 1e6


class OpenAIBackend:
    """OpenAI Responses API (e.g. gpt-6-astra), chained with previous_response_id like upstream's `codex exec resume`.
    Reasoning summaries are "detailed", as in upstream's Codex config."""

    def __init__(self, model: str, effort: str = "xhigh", system: str = INSTRUCTIONS, timeout_s: float = 3600):
        import openai
        if model not in OPENAI_PRICES:
            raise ValueError(f"no price table for {model}; add it to OPENAI_PRICES in backends.py")
        self.client = openai.OpenAI(max_retries=8, timeout=timeout_s)
        self.model, self.effort, self.system = model, effort, system
        self.previous: str | None = None

    def send(self, user_text: str) -> Reply:
        t0 = time.time()
        r = self.client.responses.create(model=self.model, instructions=self.system, input=user_text,
                                         reasoning={"effort": self.effort, "summary": "detailed"},
                                         previous_response_id=self.previous)
        self.previous = r.id
        reasoning = "\n\n".join(s.text for item in r.output if item.type == "reasoning" for s in (item.summary or []))
        u = r.usage
        cached = getattr(u.input_tokens_details, "cached_tokens", 0) or 0
        usage = dict(input=u.input_tokens - cached, output=u.output_tokens, cache_read=cached, cache_write=0)
        return Reply(text=r.output_text, reasoning=reasoning, stop_reason=r.status, model=r.model, usage=usage,
                     usd=openai_usd(self.model, u.input_tokens, cached, u.output_tokens), seconds=round(time.time() - t0, 1), raw={"id": r.id})


def make_backend(cfg: dict):
    cfg = dict(cfg)
    provider = cfg.pop("provider")
    if provider == "anthropic":
        return AnthropicBackend(**cfg)
    if provider == "openai":
        return OpenAIBackend(**cfg)
    raise ValueError(f"unknown backend provider {provider!r}; use 'anthropic' or 'openai'")
