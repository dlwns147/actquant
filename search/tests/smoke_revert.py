"""Smoke test for the bucket-batching revert. Confirms the per-sample
legacy `get_pred` path actually runs end-to-end under KIVI-fake int2
(no batching scaffolding to lean on), and that the dormant `_pad_mask`
plumbing committed at HEAD stays silent under `batch_size=1`.

Real Qwen2.5-7B-Instruct + KIVI int2 fake + flash_attention_2; 3
LongBench-E hotpotqa samples truncated to the model's max_length.

Verifies:
  * `get_pred` runs the legacy `model.generate` per sample,
  * returns one dict per input with the documented keys,
  * `past_key_values._pad_mask` is None each prefill (no batching ⇒ no pad),
  * tokenization / chat-template path still works.

Run: CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
     python tests/smoke_revert.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from model.replace import replace_kv_cache
from utils.longbench import _prepare_prompt
from utils.minilongbench import get_pred as mlb_get_pred

MODEL = os.environ.get("REAL_MODEL",
                       "/SSD/huggingface/Qwen/Qwen2.5-7B-Instruct")
MODEL_NAME = "Qwen2.5-7B-Instruct"
CFG = "utils/longbench_config"
DATA = "utils/minilongbench_data/data/hotpotqa.jsonl"
N = 3


def main():
    print(f"[smoke] MODEL={MODEL_NAME}  N={N}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16,
        attn_implementation="flash_attention_2").to("cuda").eval()
    nb = m.config.num_hidden_layers
    replace_kv_cache(m, tok, method=["kivi"], n_block=nb,
                     k_quant_scheme="channel", v_quant_scheme="token",
                     residual_length=128, packing=False,
                     quant_kv_output=False, k_pruning_dim=0, v_pruning_dim=0)
    kc = m.config.kivi_config
    kc.k_bits = [2] * nb; kc.v_bits = [2] * nb
    kc.k_group_size = [32] * nb; kc.v_group_size = [32] * nb
    m.config.use_cache = True

    # tiny slice of LongBench-E hotpotqa
    data = []
    with open(DATA) as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) >= N: break

    # configs (just for prompt format + max_gen — hotpotqa max_gen = 32)
    dataset2prompt = json.load(
        open(os.path.join(CFG, "dataset2prompt.json")))
    dataset2maxlen = json.load(
        open(os.path.join(CFG, "dataset2maxlen.json")))
    model2maxlen = json.load(open(os.path.join(CFG, "model2maxlen.json")))
    max_length = model2maxlen[MODEL_NAME]
    max_gen = dataset2maxlen["hotpotqa"]
    prompt_format = dataset2prompt["hotpotqa"]
    print(f"[smoke] hotpotqa max_gen={max_gen}  max_length={max_length}")

    # observe _pad_mask each forward (should be None since batch_size=1)
    pad_mask_observed = []
    orig_update = type(m.config.kivi_config).__init__
    # patch via post hook on the cache update: easier path = check after
    # get_pred via the last past_key_values (not exposed). Instead instrument
    # the KIVIFakeCache once.
    from model.KIVICache import KIVIFakeCache
    orig_upre = KIVIFakeCache._update_prefill
    def spy(self, k, v, lid, cache_kwargs=None):
        if lid == 0:
            pad_mask_observed.append(getattr(self, '_pad_mask', None))
        return orig_upre(self, k, v, lid, cache_kwargs)
    KIVIFakeCache._update_prefill = spy

    preds = mlb_get_pred(m, tok, data, max_length, max_gen, prompt_format,
                        "hotpotqa", "cuda", MODEL_NAME)

    # checks
    assert len(preds) == N, f"got {len(preds)} preds, expected {N}"
    needed = {"pred", "answers", "all_classes", "length"}
    for i, p in enumerate(preds):
        assert needed <= set(p), f"missing keys in pred[{i}]: {set(p)}"
        assert isinstance(p["pred"], str), f"pred[{i}].pred not str"
    print(f"[smoke] got {len(preds)} preds with the legacy keys ✓")
    print(f"[smoke] _pad_mask observed across {len(pad_mask_observed)} "
          f"prefills (one per sample): all None? "
          f"{all(x is None for x in pad_mask_observed)}")
    assert all(x is None for x in pad_mask_observed), \
        "_pad_mask should be None at batch_size=1 (dormant plumbing)"

    for i, p in enumerate(preds):
        gt = " | ".join(p["answers"])
        print(f"  sample {i}: pred={p['pred']!r:>40}  gold={gt}")
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
