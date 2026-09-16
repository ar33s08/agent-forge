"""Small exponential-backoff helper shared by the agent loop and providers."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RetryPolicy:
    max_attempts: int= 3
    base_delay_s: float= 0.2
    max_delay_s: float= 5.0
    backoff: float= 2.0
    # tuple of exception classes the loop retries; anything else propagates at once
    retry_on: Any= field(default_factory=lambda: (Exception,))

    def delays(self) -> list[float]:
        d, out= self.base_delay_s, []
        for _ in range(max(self.max_attempts - 1, 0)):
            out.append(min(d, self.max_delay_s))
            d *= self.backoff
        return out


class ProviderError(RuntimeError):
    """Wraps any failure raised out of a provider call."""


def run_with_retries(
    fn: Callable[[], Any],
    policy: RetryPolicy,
    *,
    sleeper: Callable[[float], None]= time.sleep,
    on_retry: Callable[[int, BaseException, float], None] | None= None,
    attempts_exhausted: str= "provider attempts exhausted",
) -> Any:
    last: BaseException | None= None
    delays= policy.delays()
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return fn()
        except policy.retry_on as exc:
            last= exc
            if attempt>= policy.max_attempts:
                break
            delay= delays[attempt - 1]
            if on_retry:
                on_retry(attempt, exc, delay)
            sleeper(delay)
    raise ProviderError(f"{attempts_exhausted}: {last}") from last
