import asyncio
import functools
import inspect
from contextlib import asynccontextmanager, contextmanager
from contextvars import ContextVar
import os
import sys
from langchain_community.callbacks.openai_info import OpenAICallbackHandler
from langchain_core.tracers.context import register_configure_hook
from collections import defaultdict
from dataclasses import dataclass
from langchain_community.callbacks import OpenAICallbackHandler
from typing import Any, AsyncIterator
from langchain_core.outputs import LLMResult


_CALLBACK: ContextVar[OpenAICallbackHandler | None] = ContextVar(
    "cost_tracking_callback", default=None
)
_CHECKPOINTS: ContextVar[set[str] | None] = ContextVar("cost_checkpoint", default=None)

register_configure_hook(_CALLBACK, True)


@dataclass
class OpenAIByModelCostTracker(OpenAICallbackHandler):
    """Callback Handler that tracks OpenAI info."""

    model_trackers: dict[str, OpenAICallbackHandler]
    checkpoint_trackers: dict[str, OpenAICallbackHandler]

    def __init__(self) -> None:
        super().__init__()
        self.model_trackers = defaultdict(lambda: OpenAICallbackHandler())
        self.checkpoint_trackers = defaultdict(lambda: OpenAICallbackHandler())

    def __repr__(self) -> str:
        s = ""
        total = 0.0
        for model, tracker in self.model_trackers.items():
            s += f"Model: {model} = {tracker.total_cost}\n"
            total += tracker.total_cost
        for checkpoint, tracker in self.checkpoint_trackers.items():
            s += f"Checkpoint: {checkpoint} = {tracker.total_cost}\n"
        s += "Total Cost: " + str(total)
        return s

    @property
    def total_cost(self) -> float:  #  type: ignore
        return sum(tracker.total_cost for tracker in self.model_trackers.values())

    def _child_dict(self, child: OpenAICallbackHandler) -> dict[str, Any]:
        return {
            "total_cost": child.total_cost,
            "successful_requests": child.successful_requests,
            "completion_tokens": child.completion_tokens,
            "prompt_tokens": child.prompt_tokens,
            "total_tokens": child.total_tokens,
        }

    def dict(self) -> dict[str, Any]:
        return {
            "models": {k: self._child_dict(v) for k, v in self.model_trackers.items()},
            "checkpoints": {
                k: self._child_dict(v) for k, v in self.checkpoint_trackers.items()
            },
            "total": self.total_cost,
        }

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        if response.llm_output is None:
            return None

        if "token_usage" not in response.llm_output:
            with self._lock:
                self.successful_requests += 1
            return None

        model_name = response.llm_output.get("model_name", "")
        self.model_trackers[model_name].on_llm_end(response, **kwargs)

        checkpoints = _CHECKPOINTS.get() or set()
        for checkpoint in checkpoints:
            self.checkpoint_trackers[checkpoint].on_llm_end(response, **kwargs)


def report_costs(cb: OpenAIByModelCostTracker):
    print()
    print("Model Costs:")
    for model, tracker in cb.model_trackers.items():
        print(f"  {model} = ${tracker.total_cost:.4f}")
    print("Step Cost:")
    for checkpoint, tracker in cb.checkpoint_trackers.items():
        pct = tracker.total_cost / cb.total_cost * 100
        print(f"  {checkpoint} = ${tracker.total_cost:.4f} ({pct:.2f}%)")
    print(f"** Total Cost: ${cb.total_cost:.4f}")


@contextmanager
def set_cost_checkpoint(checkpoint_name: str):
    checkpoints = _CHECKPOINTS.get() or set()
    checkpoints.add(checkpoint_name)
    _CHECKPOINTS.set(checkpoints)

    try:
        yield
    finally:
        (_CHECKPOINTS.get() or set()).remove(checkpoint_name)


@contextmanager
def checkpoint_cost_tracking():
    cb = OpenAIByModelCostTracker()
    _CALLBACK.set(cb)
    try:
        yield cb
    finally:
        _CALLBACK.set(None)


@asynccontextmanager
async def enforce_budget(budget: float, cb: OpenAIByModelCostTracker):
    done_event = asyncio.Event()

    async def _budget_monitor():
        while not done_event.is_set():
            await asyncio.sleep(0.1)
            if cb.total_cost > budget:
                done_event.set()
                sys.stderr.write(
                    "\n\n!!!!!!!!!!! ERROR: Budget exceeded: ${:.4f}, forcefully shutting down\n\n".format(
                        budget
                    )
                )
                sys.stderr.write("------- Cost Breakdown -------\n")
                sys.stderr.write(str(cb))
                sys.stderr.write("\n------------------------------\n")
                os._exit(1)

    try:
        budget_task = None
        if budget > 0:
            budget_task = asyncio.create_task(_budget_monitor())
        yield
    finally:
        if budget_task:
            done_event.set()
            await budget_task


def cost_checkpoint(checkpoint_name: str):
    def decorator(func):
        async def _generator_passthrough(
            g: AsyncIterator, checkpoint_name: str
        ) -> AsyncIterator:
            with set_cost_checkpoint(checkpoint_name):
                async for x in g:
                    yield x

        @functools.wraps(func)
        async def _async_wrapper(*args, **kwargs):
            with set_cost_checkpoint(checkpoint_name):
                r = await func(*args, **kwargs)

                # if this is a generator, we need to consume it to track costs
                if inspect.isasyncgen(r):
                    return _generator_passthrough(r, checkpoint_name)
                else:
                    return r

        @functools.wraps(func)
        def _sync_wrapper(*args, **kwargs):
            with set_cost_checkpoint(checkpoint_name):
                r = func(*args, **kwargs)

                if inspect.isasyncgen(r):
                    return _generator_passthrough(r, checkpoint_name)
                else:
                    return r

        @functools.wraps(func)
        def _generator_wrapper(*args, **kwargs):
            with set_cost_checkpoint(checkpoint_name):
                yield from func(*args, **kwargs)

        @functools.wraps(func)
        async def _async_gen_wrapper(*args, **kwargs):
            with set_cost_checkpoint(checkpoint_name):
                async for x in func(*args, **kwargs):
                    yield x

        if inspect.isgeneratorfunction(func):
            return _generator_wrapper
        elif inspect.isasyncgenfunction(func):
            return _async_gen_wrapper
        elif asyncio.iscoroutinefunction(func):
            return _async_wrapper
        else:
            return _sync_wrapper

    return decorator
