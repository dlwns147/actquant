"""Backward-compat shim.

LlamaThinKSearchSpace has been merged into the single unified
``search_space.llama.LlamaSearchSpace`` (which now covers weights + KV group-size
+ ThinK KV-pruning + QEFT outlier columns). This module re-exports it under the
old name so existing imports keep working.
"""

from .llama import LlamaSearchSpace

# old name -> unified class
LlamaThinKSearchSpace = LlamaSearchSpace
