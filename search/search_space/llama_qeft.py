"""Backward-compat shim.

The QEFT outlier-column weight-search space has been consolidated into the
single ``search_space.llama`` module. ``LlamaQEFTSearchSpace`` and its helpers
(``build_w_options``, ``DEFAULT_QEFT_COLUMNS``, ``OUTLIER_LINEARS``) now live in
``search_space/llama.py``; this module re-exports them under the old path so
existing imports keep working.
"""

from .llama import (
    LlamaQEFTSearchSpace,
    build_w_options,
    DEFAULT_QEFT_COLUMNS,
    OUTLIER_LINEARS,
)

__all__ = [
    'LlamaQEFTSearchSpace',
    'build_w_options',
    'DEFAULT_QEFT_COLUMNS',
    'OUTLIER_LINEARS',
]
