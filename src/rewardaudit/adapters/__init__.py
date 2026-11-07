"""Adapters turn any environment or scoring function into the Adapter interface."""

from __future__ import annotations

from typing import Any

from rewardaudit.adapters.base import Adapter


def build_adapter(spec: dict[str, Any]) -> Adapter:
    """Construct an adapter from a JSON-serialisable spec. Called in the worker process."""
    kind = spec.get("kind", "function")
    if kind == "function":
        from rewardaudit.adapters.function_adapter import FunctionAdapter

        return FunctionAdapter(spec)
    if kind == "demo":
        from rewardaudit.demo import DemoAdapter

        return DemoAdapter(spec)
    if kind == "verifiers":
        from rewardaudit.adapters.verifiers_adapter import VerifiersAdapter

        return VerifiersAdapter(spec)
    raise ValueError(f"unknown adapter kind: {kind!r}")


__all__ = ["Adapter", "build_adapter"]
