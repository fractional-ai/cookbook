import asyncio
import contextlib
import functools
import inspect
import sys
import traceback
from typing import AsyncIterator, Generator
from langchain.callbacks import tracing_v2_enabled
from langsmith import trace
from langsmith.run_helpers import _PARENT_RUN_TREE
from langchain_core.tracers.context import tracing_v2_callback_var


@contextlib.contextmanager
def tracing_v2_disabled() -> Generator[None, None, None]:
    cb = tracing_v2_callback_var.get()
    parent = _PARENT_RUN_TREE.get()
    try:
        tracing_v2_callback_var.set(None)
        _PARENT_RUN_TREE.set(None)
        yield None
    finally:
        tracing_v2_callback_var.set(cb)
        _PARENT_RUN_TREE.set(parent)


@contextlib.contextmanager
def _setup_tracing(name: str, enabled: bool):
    if not enabled:
        with tracing_v2_disabled():
            yield
    else:
        with tracing_v2_enabled(), trace(name=name) as t:
            try:
                yield t
            except Exception as e:
                t.error = format_exception_for_langsmith(e)
                raise


def format_exception_for_langsmith(e: Exception) -> str:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    formatted_trace = "".join(
        traceback.format_exception(exc_type, exc_value, exc_traceback)
    )

    return f"{e.__class__.__name__}: {e}\n\n{formatted_trace}"


def configure_tracing(
    name: str | None = None,
    include_inputs: bool = True,
    enabled: bool = True,
    *,
    metadata: dict[str, str] = {},
    **kwargs,
):
    """Decorator for configuring tracing for a function."""
    trace_kwargs = {"metadata": metadata, **kwargs}

    if not include_inputs:
        trace_kwargs["process_inputs"] = lambda _: {}

    def decorator(func):
        fn = func

        async def _generator_passthrough(
            g: AsyncIterator, name: str, enabled: bool
        ) -> AsyncIterator:
            with _setup_tracing(name, enabled) as t:
                if t:
                    t.outputs = {"generator_output": []}

                async for x in g:
                    if t:
                        t.outputs["generator_output"].append(x)
                    yield x

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _setup_tracing(name, enabled) as t:
                outputs = fn(*args, **kwargs)

                # if this is a generator, we need to consume it to keep the correct racing context
                if inspect.isasyncgen(outputs):
                    return _generator_passthrough(outputs, name, enabled)
                else:
                    if t:
                        t.outputs = {"output": outputs}
                    return outputs

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            with _setup_tracing(name, enabled) as t:
                outputs = await fn(*args, **kwargs)

                # if this is a generator, we need to consume it to keep the correct racing context
                if inspect.isasyncgen(outputs):
                    return _generator_passthrough(outputs, name, enabled)
                else:
                    if t:
                        t.outputs = {"output": outputs}
                    return outputs

        @functools.wraps(func)
        def generator_wrapper(*args, **kwargs):
            with _setup_tracing(name, enabled):
                yield from func(*args, **kwargs)

        @functools.wraps(func)
        async def async_gen_wrapper(*args, **kwargs):
            with _setup_tracing(name, enabled):
                async for x in func(*args, **kwargs):
                    yield x

        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        elif inspect.isgeneratorfunction(func):
            return generator_wrapper
        elif inspect.isasyncgenfunction(func):
            return async_gen_wrapper
        else:
            return wrapper

    return decorator
