"""Put the repo root on sys.path so `pytest tests/` and
`python tests/<file>.py` both resolve the top-level modules
(evaluator, post_search, search, utils.*) without a per-file shim.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
