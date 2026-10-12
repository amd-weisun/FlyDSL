# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Device code of the Qwen3.5 decode MonoKernel.

``build.py`` builds the one launch from the per-stage modules (``gdn`` / ``attn``, ``mixer``, ``ffn``) over the helpers
``common.py`` binds; ``plan.py`` holds the execution constants, per-stage task counts and mailbox
layout the host side shares.
"""

from kernels.monokernel.qwen35.kernel.build import build_qwen35_kernel

__all__ = ["build_qwen35_kernel"]
