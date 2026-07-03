"""GP backend registry.

`get_backend(name)` returns a GPBackend implementation. The concrete backend module
(and its heavy gpflow/TF imports) is imported lazily on first use, so importing
`emcal` does not require gpflow/TensorFlow until a GP model is actually built.
"""
from .base import GPBackend

_BACKENDS = ("gpflow",)


def get_backend(name="gpflow"):
    """Return a GPBackend instance for `name` (default 'gpflow')."""
    key = (name or "gpflow").lower()
    if key == "gpflow":
        from .gpflow_backend import GpflowBackend

        return GpflowBackend()
    raise ValueError(
        f"Unknown GP backend {name!r}. Available backends: {_BACKENDS}."
    )


__all__ = ["GPBackend", "get_backend"]
