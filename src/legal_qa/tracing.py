"""Optional Langfuse tracing. Active only when LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY are set
(LANGFUSE_HOST for self-hosted; defaults to Langfuse Cloud). Never breaks the request path."""
import os
from contextlib import contextmanager


def enabled() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))


def _client():
    from langfuse import get_client
    return get_client()


@contextmanager
def trace(name: str, **input_):
    """Yields a dict; fill ctx['output'] and ctx['metadata']; ctx['trace_id'] is set when tracing is on."""
    ctx: dict = {"trace_id": None}
    if not enabled():
        yield ctx
        return
    try:
        lf = _client()
        cm = lf.start_as_current_span(name=name, input=input_)
        span = cm.__enter__()
        ctx["trace_id"] = lf.get_current_trace_id()
    except Exception as e:  # noqa: BLE001
        print("langfuse disabled for this request:", e)
        yield ctx
        return
    try:
        yield ctx
    finally:
        try:
            span.update(output=ctx.get("output"), metadata=ctx.get("metadata"))
            cm.__exit__(None, None, None)
        except Exception as e:  # noqa: BLE001
            print("langfuse flush error:", e)


def score(trace_id: str | None, name: str, value: float) -> None:
    if not (enabled() and trace_id):
        return
    try:
        lf = _client()
        lf.create_score(trace_id=trace_id, name=name, value=float(value))
        lf.flush()
    except Exception as e:  # noqa: BLE001
        print("langfuse score error:", e)
