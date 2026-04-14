"""Thin OpenAI-compatible chat client.

- Heartbeat logging while a call is in flight.
- Auto-switches between ``max_tokens`` and ``max_completion_tokens``.
- Auto-omits temperature if a provider rejects it.
- Extracts the longest Python code block from a response.
"""

from __future__ import annotations

import logging
import re
import threading
import time

from openai import OpenAI, APIError

from .config import LLMConfig

logger = logging.getLogger("mle-solver")


class _Heartbeat:
    def __init__(self, label: str, interval: float = 30.0):
        self.label = label
        self.interval = interval
        self._stop = threading.Event()
        self._t: threading.Thread | None = None
        self._started: float = 0.0

    def __enter__(self):
        self._started = time.time()
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2)

    def _run(self):
        while not self._stop.wait(self.interval):
            logger.info(f"[llm] {self.label} still waiting {time.time() - self._started:.0f}s...")


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.client = OpenAI(
            api_key=cfg.api_key or "missing-key",
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=0,  # we handle retries ourselves
        )
        self._legacy_max_tokens = False
        self._omit_temperature = False

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        max_tokens: int | None = None,
        label: str = "chat",
    ) -> str:
        temp = self.cfg.temperature if temperature is None else temperature
        max_t = self.cfg.max_tokens if max_tokens is None else max_tokens
        started = time.time()
        last_err: Exception | None = None

        for attempt in range(self.cfg.max_retries):
            try:
                content = self._call(messages, temp, max_t, label, reasoning_effort)
                logger.info(f"[llm] <- {label} OK in {time.time() - started:.0f}s chars={len(content)}")
                return content
            except APIError as e:
                err = str(e)
                err_l = err.lower()
                if "max_tokens" in err and "max_completion_tokens" in err:
                    self._legacy_max_tokens = not self._legacy_max_tokens
                    continue
                if "temperature" in err_l and any(k in err_l for k in ("does not support", "only supports", "unsupported")):
                    self._omit_temperature = True
                    continue
                if "reasoning_effort" in err_l and reasoning_effort is not None:
                    logger.warning(f"[llm] {label} reasoning_effort unsupported, retrying without it")
                    reasoning_effort = None
                    continue
                if "rate_limit_error" in err_l or "429" in err_l:
                    reduced = max(512, min(2048, max_t // 2))
                    if reduced < max_t:
                        logger.warning(
                            f"[llm] {label} hit rate limit; reducing max_tokens {max_t} -> {reduced}"
                        )
                        max_t = reduced
                last_err = e
                wait = min(10 * 2 ** attempt, 60)
                logger.warning(
                    f"[llm] {label} attempt {attempt+1}/{self.cfg.max_retries} "
                    f"failed: {type(e).__name__}: {e} — sleeping {wait}s"
                )
                time.sleep(wait)
            except RuntimeError as e:
                last_err = e
                wait = min(10 * 2 ** attempt, 60)
                logger.warning(
                    f"[llm] {label} attempt {attempt+1}/{self.cfg.max_retries} "
                    f"RuntimeError: {e} — sleeping {wait}s"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"[llm] {label} failed after {self.cfg.max_retries} attempts "
            f"in {time.time() - started:.0f}s: {last_err}"
        )

    def _call(
        self,
        messages,
        temperature,
        max_tokens,
        label,
        reasoning_effort: str | None,
    ):
        kwargs = (
            {"max_completion_tokens": max_tokens}
            if not self._legacy_max_tokens
            else {"max_tokens": max_tokens}
        )
        effort = (reasoning_effort or "").strip().lower()
        if effort:
            kwargs["reasoning_effort"] = effort
        if not self._omit_temperature and temperature is not None and effort in {"", "none"}:
            kwargs["temperature"] = temperature
        with _Heartbeat(label=label):
            resp = self.client.chat.completions.create(
                model=self.cfg.model, messages=messages, **kwargs  # type: ignore[arg-type]
            )
        if not resp.choices or not resp.choices[0].message.content:
            raise RuntimeError("LLM returned no content")
        return resp.choices[0].message.content

    # ── code extraction ───────────────────────────────────────────────────

    _FENCED = re.compile(r"```\s*(?:python|py|python3)?\s*\n(.*?)\n\s*```", re.DOTALL | re.IGNORECASE)
    _PERMISSIVE = re.compile(r"```[^\n`]*?\n?(.*?)```", re.DOTALL)

    def extract_python_code(self, text: str) -> str:
        if not text:
            return ""
        for pattern in (self._FENCED, self._PERMISSIVE):
            blocks = pattern.findall(text)
            if blocks:
                substantive = [b for b in blocks if len(b.strip()) >= 30]
                return max(substantive or blocks, key=len).strip()
        return ""
