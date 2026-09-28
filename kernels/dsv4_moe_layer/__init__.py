from __future__ import annotations

from typing import TYPE_CHECKING

from kernels.dsv4_moe_layer.config import MoeMode

if TYPE_CHECKING:
    from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

__all__ = ["MoeMode", "Dsv4MoeLayer"]


def __getattr__(name: str):
    """Load the GPU wrapper only when callers request it."""

    if name == "Dsv4MoeLayer":
        from kernels.dsv4_moe_layer.layer import Dsv4MoeLayer

        return Dsv4MoeLayer
    raise AttributeError(name)
