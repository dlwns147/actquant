"""Smoke test: AWQ on google/gemma-3-12b-it (text decoder).

Verifies the Gemma3 wiring end-to-end (get_blocks / move_embed /
auto_scale Gemma3 branch / scale_ln_fcs_gemma routing / apply) and that
4-bit AWQ does NOT catastrophically degrade the model.
"""
import sys, gc, torch
sys.path.insert(0, "/NAS/SJ/actquant/search")
from quant.awq import AWQ

MODEL = "/SSD/huggingface/google/gemma-3-12b-it"
N_BLOCK = 48
LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
           "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
arch = {l: [4] * N_BLOCK for l in LINEARS}


from transformers import AutoTokenizer
EVAL_TOK = AutoTokenizer.from_pretrained(MODEL)   # fast tokenizer for trustworthy eval


@torch.no_grad()
def ppl(model, n_seg=10, seqlen=512):
    from datasets import load_dataset
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    enc = EVAL_TOK("\n\n".join(data["text"]), return_tensors="pt").input_ids
    dev = next(model.parameters()).device
    nlls, n = [], min(n_seg, enc.shape[1] // seqlen)
    for i in range(n):
        ids = enc[:, i * seqlen:(i + 1) * seqlen].to(dev)
        out = model(ids, labels=ids)
        nlls.append(out.loss.float() * seqlen)
    return torch.exp(torch.stack(nlls).sum() / (n * seqlen)).item()


@torch.no_grad()
def sample_gen(model):
    msgs = [{"role": "user", "content": "In one sentence, what is the capital of France?"}]
    ids = EVAL_TOK.apply_chat_template(msgs, add_generation_prompt=True,
                                       return_tensors="pt").to(next(model.parameters()).device)
    out = model.generate(ids, max_new_tokens=32, do_sample=False)
    return EVAL_TOK.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    m = AWQ(model_name=MODEL, config={}, arch=arch, device_map={"": 0},
            group_size=128, dtype=torch.bfloat16, clip_asym=True)

    print(f"model class: {m.model.__class__.__name__}")
    from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
    from quant.awq_utils.pre_quant import get_blocks
    blocks = get_blocks(m.model)
    assert len(blocks) == N_BLOCK, f"expected {N_BLOCK} blocks, got {len(blocks)}"
    assert isinstance(blocks[0], Gemma3DecoderLayer), type(blocks[0])
    print(f"get_blocks OK: {len(blocks)} Gemma3DecoderLayer")

    m.model = m.model.to("cuda:0")
    base = ppl(m.model)
    print(f"[baseline FP16] wikitext2 ppl = {base:.4f}")
    print(f"[baseline FP16] gen: {sample_gen(m.model)!r}")
    m.model = m.model.to("cpu")
    gc.collect(); torch.cuda.empty_cache()

    m.run(nsamples=8, seqlen=512)            # AWQ scale + clip + 4-bit pseudo-quant

    q = ppl(m.model)
    qgen = sample_gen(m.model)
    print(f"[AWQ 4bit ] wikitext2 ppl = {q:.4f}")
    print(f"[AWQ 4bit ] gen: {qgen!r}")
    ratio = q / base
    print(f"ppl ratio (awq/fp16) = {ratio:.3f}")
    # The documented Gemma3 AWQ bug (generic RMSNorm scale absorption ignoring
    # the (1+w) offset) blows ppl up by 1-2 orders of magnitude AND produces
    # incoherent generation. A correct 4-bit AWQ with a tiny 8-sample, pure
    # 4-bit (no mixed precision) calibration lands well under ~3x and keeps
    # generation coherent. The coherence check is the strongest signal since
    # the absolute ppl is inflated (instruction-tuned model on raw wikitext).
    assert ratio < 3.0, f"AWQ degraded catastrophically (ratio={ratio:.2f})"
    assert "paris" in qgen.lower(), f"AWQ generation incoherent: {qgen!r}"
    print("SMOKE PASS")


if __name__ == "__main__":
    main()
