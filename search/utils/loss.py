import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import ast
import os
import json
import hashlib

# class JSD(nn.Module):
#     def __init__(self, reduction='batchmean'):
#         super(JSD, self).__init__()
#         self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)

#     def forward(self, p: torch.tensor, q: torch.tensor):
#         p, q = p.log_softmax(-1), q.log_softmax(-1)
#         m = (0.5 * (p + q))
#         return 0.5 * (self.kl(m, p) + self.kl(m, q))

class JSD(nn.Module):
    def __init__(self, reduction='batchmean', eps=1e-7):
        super(JSD, self).__init__()
        self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)
        self.eps = eps

    def forward(self, p: torch.tensor, q: torch.tensor):
        m = (0.5 * (p.softmax(-1) + q.softmax(-1))).clamp_min(self.eps).log()
        return 0.5 * (self.kl(m, p.log_softmax(-1)) + self.kl(m, q.log_softmax(-1)))


class ForwardKL(nn.Module):
    """Directional forward KL( teacher ‖ student ) = KL(FP16 ‖ candidate).
    eval_loss calls forward(p=candidate_logits, q=FP16_logits) -> teacher is q.
    KLDivLoss(input=log student, target=log teacher, log_target=True)
      = sum softmax(teacher)*(log_softmax(teacher) - log_softmax(student)) = KL(Q‖P).
    Downstream-aware proxy study (2026-06): beats symmetric JSD for final SELECTION
    on Llama-3.1-8B (EntropyGatedJSD on Qwen2.5-7B). LOWER IS BETTER (== JSD)."""
    def __init__(self, reduction='batchmean', eps=1e-7):
        super(ForwardKL, self).__init__()
        self.kl = nn.KLDivLoss(reduction=reduction, log_target=True)

    def forward(self, p: torch.tensor, q: torch.tensor):
        return self.kl(p.log_softmax(-1), q.log_softmax(-1))


def TopK(p: torch.tensor, q: torch.tensor, k: int):
    p_topk, q_topk = p.topk(k, dim=-1, largest=True), q.topk(k, dim=-1, largest=True)
    pq = torch.cat((p_topk, q_topk), dim=-1)
    union, counts = pq.unique(dim=-1, return_inverse=False, return_counts=True)
    intersection = pq[torch.where(counts.gt(1))]
    return (intersection / union).mean()

def get_key_token_list(
    evaluator_model, 
    evaluator_tokenizer, 
    loader, 
    tokenizer=None,
    save_path='', 
    load_path='', 
    trunc_len=4096, 
    sliding_window=1024, 
    alpha=2, 
    beta=-2,
    mode='offline',
    verbosity=False,
    split='train',
    manifest_meta=None,
    doc_spans=None,
    resume=False
):
    """
    Get key token list from loader.
    
    Parameters:
        evaluator_model: Model used to identify key tokens (for online mode)
        evaluator_tokenizer: Tokenizer for evaluator model
        loader: DataLoader containing input data
        tokenizer: Tokenizer for the evaluated model (optional, for decode if needed)
        save_path: Path to save key tokens (for online mode)
        load_path: Path to load precomputed key tokens (for offline mode)
        trunc_len: Length of truncated short context
        sliding_window: Size of sliding window
        alpha: Threshold for LSD
        beta: Threshold for LCL
        mode: 'online' to compute key tokens, 'offline' to use precomputed
        verbosity: If True, print decoded key tokens for debugging
        doc_spans: Optional token range(s) the DOCUMENT body occupies inside a
            sample -- `(start, end)`, or `[(s0,e0), (s1,e1), ...]` when the
            document is SPLIT (the `chat:` layout puts the assistant header
            between context and tail). The same ranges apply to every sample.
            Needed when
            the loader wraps the document in a chat template ('chat:<corpus>'):
            the decode / offsets / cal_overlap then run over the document slice
            ONLY, so the text matches the archive exactly (manifest 'exact', not
            'prefix'), and the returned indices are shifted back into
            full-sequence coordinates. Without it the model-specific template
            boilerplate that survives skip_special_tokens (measured: 0 chars
            Mistral-v0.3, 5 Gemma-3, 77 Llama-3.1, 81 Qwen2.5) would shift every
            character offset and the mask would land on the wrong tokens.
    
    Returns:
        List of key token indices per batch: [batch_idx][seq_idx] -> list of token indices
    """
    key_token_list = []
    if tokenizer is None:
        tokenizer = evaluator_tokenizer
    # `tokenizer` is the TARGET's: it decodes the loader ids and supplies the
    # offset_mapping the intervals are mapped back onto. `evaluator_tokenizer`
    # is only used inside find_key_token, which re-tokenizes the TEXT — the two
    # models never exchange token ids (that is why LongPPL passes character
    # spans in the first place).
    _texts = []
    _n_iv = []
    _slice_base = 0        # running global sample index (uneven last batch)
    # Number of DOCUMENTS the loader will yield. len(loader) counts BATCHES,
    # so comparing it against the archive's n_slices falsely rejects every
    # valid archive as soon as data_batch_size > 1.
    _n_docs = None
    _ds = getattr(loader, 'dataset', None)
    if _ds is not None:
        try:
            _n_docs = len(_ds)
        except TypeError:
            _n_docs = None
    _manifest = None
    if mode == 'offline' and load_path:
        _mpath = os.path.join(load_path, KEY_TOKEN_MANIFEST)
        if os.path.exists(_mpath):
            with open(_mpath) as _f:
                _manifest = json.load(_f)
        else:
            print(f"[key_token] {load_path} has no {KEY_TOKEN_MANIFEST}: the "
                  f"intervals cannot be checked against the documents this "
                  f"loader produced (pre-manifest archive). Regenerate with "
                  f"gen_key_token.py to make it verifiable.")
    # Content-addressed slice lookup. Positional indexing (document i -> slice_i)
    # is only right when this process walks the whole loader in order, and
    # accelerator.prepare() breaks exactly that: MEASURED at num_processes=2,
    # process 0 receives documents [0, 2] and process 1 [1, 3], while BOTH count
    # their slices from 0 — so half the documents get another document's key
    # tokens. Hashing the decoded text picks the right slice regardless of which
    # shard, order, or batch size this process sees. Positional indexing stays
    # as the fallback for legacy archives and for the 'prefix' case (evaluator-
    # truncated text, whose hash cannot match by construction).
    _shard_remap = 0
    _reused = 0
    _prefix_n = 0
    _sha2idx = {}
    if _manifest:
        for _i, _rec in enumerate(_manifest.get('slices') or []):
            _sha2idx.setdefault(_rec['sha256'], _i)
    for batch_idx, (inputs, attention_mask, labels) in enumerate(loader):
        batch_key_tokens = []
        batch_size = inputs.shape[0]
        
        for seq_idx in range(batch_size):
            # Get actual input_ids (remove padding)
            slice_idx = _slice_base + seq_idx
            input_ids = inputs[seq_idx:seq_idx+1]
            if attention_mask is not None:
                mask = attention_mask[seq_idx]
                actual_length = mask.sum().item()
                input_ids = input_ids[:, :actual_length]

            # Chat-templated loaders: restrict every downstream step (decode,
            # offsets, cal_overlap) to the DOCUMENT body, then map the indices
            # back. `_posmap` is None for the plain corpora -> exact no-op.
            _posmap = None
            if doc_spans is not None:
                _rng = _norm_doc_spans(doc_spans)
                _posmap = torch.cat([torch.arange(s, e) for s, e in _rng])
                # gathering the ids of every range and decoding THAT reproduces
                # the archive's text exactly, because a split document's parts
                # are contiguous in the original (the header is inserted, not
                # substituted) -- so the manifest still matches 'exact'.
                input_ids = torch.cat([input_ids[:, s:e] for s, e in _rng], dim=1)
            
            # For offline mode, try to load from file
            if mode == 'offline':
                # slice_path = os.path.join(load_path, f"batch_{batch_idx}_seq_{seq_idx}.txt")\
                assert os.path.exists(load_path), (
                    f"key-token archive not found: {load_path}")
                # Need to decode to get text for offset mapping
                text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                offset_mapping = _loader_offsets(text, tokenizer, input_ids)

                # which slice file belongs to THIS document (see _sha2idx above)
                file_idx = _sha2idx.get(_text_sha(text), slice_idx)
                if file_idx != slice_idx:
                    _shard_remap += 1
                    # NOTE: doc_spans used to be a PER-SLICE list, which is
                    # positional and therefore unusable once the hash lookup
                    # proves the loader is out of archive order — that case had
                    # to be refused. The contract is now "the range(s) the
                    # document occupies", IDENTICAL for every sample (the chat
                    # affixes are a fixed-length prefix/suffix), so it carries no
                    # positional information and a remapped slice is harmless.

                slice_path = os.path.join(load_path, f"slice_{file_idx}.txt")
                assert os.path.exists(slice_path), (
                    f"key-token archive {load_path} has no slice_{file_idx}.txt "
                    f"-- the loader is asking for more documents than the archive "
                    f"holds. Regenerate it with a matching n_sample/seqlen.")

                _st = check_key_token_manifest(
                    _manifest, file_idx, text, n_tokens=int(input_ids.shape[-1]),
                    n_slices=_n_docs, where=f'/{load_path}',
                    knobs=dict(trunc_len=trunc_len, sliding_window=sliding_window,
                               alpha=alpha, beta=beta))
                if _st == 'prefix':
                    _prefix_n += 1
                if _st == 'prefix' and slice_idx == 0:
                    print(f"[key_token] manifest text is a PREFIX of this "
                          f"loader's text (the archive truncated with a "
                          f"different tokenizer) — character offsets stay "
                          f"valid, accepted.")
                key_text_slices = load_key_token(slice_path)
                if key_text_slices is not None:
                    key_tokens = cal_overlap(offset_mapping, key_text_slices)
                    batch_key_tokens.append(_map_keys(key_tokens, _posmap))
                    
                    # Print decoded key tokens if verbosity is enabled
                    if verbosity and key_tokens is not None and len(key_tokens) > 0:
                        # key_tokens are indices for shift_logits (predicting token at idx+1)
                        # So actual input_ids index is idx + 1
                        key_token_ids = [input_ids[0, idx + 1].item() for idx in key_tokens]                            
                        if key_token_ids:
                            decoded_tokens = tokenizer.decode(key_token_ids, skip_special_tokens=True)
                            print(f"[Offline] [Slice {slice_idx}] {len(key_tokens)} key tokens: {decoded_tokens[:200]}")
                        else:
                            print(f"[Offline] [Slice {slice_idx}] {len(key_tokens)} key tokens (could not decode)")
                        
                else:
                    batch_key_tokens.append(None)

            elif mode == 'online':
                assert evaluator_model is not None
                # Need to decode for online mode
                text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                _texts.append(text)
                
                # slice_save_path = os.path.join(save_path, f"batch_{batch_idx}_seq_{seq_idx}.txt") if save_path else ''
                slice_save_path = os.path.join(save_path, f"slice_{slice_idx}.txt") if save_path else ''
                if resume and slice_save_path and os.path.exists(slice_save_path):
                    # every slice file is written atomically (see find_key_token),
                    # so an existing one is complete
                    key_text_slices = load_key_token(slice_save_path)
                    _reused += 1
                else:
                    key_text_slices = find_key_token(
                        text, evaluator_model, evaluator_tokenizer,
                        trunc_len, sliding_window, slice_save_path, alpha, beta
                    )
                _n_iv.append(0 if key_text_slices is None else len(key_text_slices))
                
                offset_mapping = _loader_offsets(text, tokenizer, input_ids)

                if key_text_slices is not None:
                    key_tokens = cal_overlap(offset_mapping, key_text_slices)
                    batch_key_tokens.append(_map_keys(key_tokens, _posmap))
                    
                    # Print decoded key tokens if verbosity is enabled
                    if verbosity and key_tokens is not None and len(key_tokens) > 0:
                        # key_tokens are indices for shift_logits (predicting token at idx+1)
                        # So actual input_ids index is idx + 1
                        key_token_ids = [input_ids[0, idx + 1].item() for idx in key_tokens]                             
                        if key_token_ids:
                            decoded_tokens = tokenizer.decode(key_token_ids, skip_special_tokens=True)
                            print(f"[Online] [Slice {slice_idx}] {len(key_tokens)} key tokens: {decoded_tokens[:200]}")
                        else:
                            print(f"[Online] [Slice {slice_idx}] {len(key_tokens)} key tokens (could not decode)")
                else:
                    batch_key_tokens.append(None)
            else:
                raise NotImplementedError
        
        # NESTED [batch][seq], matching how eval_loss/get_logits index this
        # (`key_token_list[batch_idx][seq_idx]`) and matching their own
        # dense_logits_list. It used to `extend` into a FLAT [slice] list, so
        # with batch_size=1 the consumer read key_token_list[b][0] — the FIRST
        # KEY TOKEN INDEX, an int — and get_loss_mask then masked exactly ONE
        # position per sequence instead of all key tokens.
        key_token_list.append(batch_key_tokens)
        _slice_base += batch_size

    # Backstop for loaders that expose no .dataset: the number of documents we
    # actually consumed must match the archive, or slice_i did not refer to
    # document i for the whole run.
    if _manifest is not None:
        _m_n = _manifest.get('n_slices')
        if _m_n and _slice_base > int(_m_n):
            raise ValueError(
                f"[key_token/{load_path}] consumed {_slice_base} documents but "
                f"the archive holds {_m_n}: the loader protocol does not match "
                f"the one the archive was generated with.")
        if _prefix_n:
            print(f"[key_token] {_prefix_n}/{_slice_base} slices matched only as "
                  f"a PREFIX: the archive was generated with the evaluator's own "
                  f"tokenization, so it covers less of each window than this "
                  f"loader produces. Regenerate with --target_model to make it "
                  f"exact.")
        if _shard_remap:
            print(f"[key_token] {_shard_remap}/{_slice_base} documents did not "
                  f"arrive in archive order (sharded or reordered loader) and "
                  f"were matched to their slice by content hash.")

    if mode == 'online' and _reused:
        print(f"[key_token] reused {_reused}/{_slice_base} slices already present "
              f"in {save_path} (--resume)")
    if mode == 'online' and save_path:
        write_key_token_manifest(
            save_path, _texts,
            dict(manifest_meta or {}, trunc_len=trunc_len,
                 sliding_window=sliding_window, alpha=alpha, beta=beta),
            n_intervals=_n_iv or None)
        _empty = sum(1 for n in _n_iv if n == 0)
        print(f"[key_token] wrote {KEY_TOKEN_MANIFEST} for {len(_texts)} slices "
              f"→ {save_path}"
              + (f"  ⚠ {_empty} slice(s) have NO key token: eval_loss SKIPS those "
                 f"documents, so the metric averages over {len(_texts) - _empty} "
                 f"of {len(_texts)}" if _empty else ""))

    _flat = [k for batch in key_token_list for k in batch]
    _none = sum(1 for k in _flat if k is None or len(k) == 0)
    if _none:
        print(f"[key_token] {_none}/{len(_flat)} documents carry no key token — "
              f"eval_loss skips them, so the metric is a mean over the remaining "
              f"{len(_flat) - _none}.")
    return key_token_list
    

def _norm_doc_spans(doc_spans):
    """`(s,e)` or `[(s0,e0), ...]` -> a list of int ranges."""
    if len(doc_spans) == 2 and all(isinstance(v, (int, np.integer)) for v in doc_spans):
        return [(int(doc_spans[0]), int(doc_spans[1]))]
    return [(int(s), int(e)) for s, e in doc_spans]


def _map_keys(key_tokens, posmap):
    """Local (document-slice) shift indices -> absolute shift indices.

    cal_overlap returns SHIFT indices: local j scores the prediction of local
    token j+1. That token is absolute `posmap[j+1]`, and the absolute shift
    index predicting it is `posmap[j+1] - 1`. For one contiguous range this is
    exactly the old `j + start`; across a split document it correctly jumps the
    gap, so a key token at the very start of the tail is scored from the last
    header position (which is where the model actually predicts it).
    """
    if posmap is None or key_tokens is None:
        return key_tokens
    n = len(posmap)
    return [int(posmap[j + 1]) - 1 for j in key_tokens if 0 <= j + 1 < n]


KEY_TOKEN_MANIFEST = 'meta.json'


def _text_sha(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def write_key_token_manifest(save_path, texts, meta, n_intervals=None):
    """Record WHAT the key-token intervals were computed on.

    The intervals are CHARACTER spans into one specific document text. Nothing
    in `slice_<i>.txt` says which document, which truncation, or which
    tokenizer produced it — the directory name is prose. Loading the wrong
    archive therefore masks the wrong tokens and still returns a number, so the
    text each slice belongs to is hashed here and checked at load time.
    """
    if not save_path:
        return
    os.makedirs(save_path, exist_ok=True)
    payload = dict(meta)
    # derived here, never taken from the caller: the sample-count check at load
    # time relies on it, and a caller that forgets it would disable that check
    # silently (the retrofit path did exactly that).
    payload['n_slices'] = len(texts)
    # n_intervals distinguishes "the evaluator found no key token here" (a real
    # outcome — that document is then SKIPPED by eval_loss, not scored in full)
    # from a truncated write: both leave a 0-byte slice file.
    payload['slices'] = [
        dict(sha256=_text_sha(t), chars=len(t),
             **({'n_intervals': int(n_intervals[i])} if n_intervals else {}))
        for i, t in enumerate(texts)]
    with open(os.path.join(save_path, KEY_TOKEN_MANIFEST), 'w') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def check_key_token_manifest(manifest, slice_idx, text, n_tokens=None,
                             n_slices=None, where='', knobs=None):
    """Does `text` match the document slice_<slice_idx> was computed on?

    Returns a short status string. Three outcomes:
      'exact'   the same text
      'prefix'  the manifest's text is a PREFIX of this one — what happens when
                the archive was generated with the evaluator's tokenizer doing
                the seqlen truncation (it cuts the same document at a different
                character), MEASURED on the shipped gov_report archives. The
                character offsets stay valid, so this is accepted and reported.
      raises    anything else: a different document, a different truncation
                direction, or a different corpus.
    """
    if not manifest:
        return 'unverified'
    # Protocol first: a text-hash PREFIX match is not enough on its own. The
    # first 2048 tokens of a document are a prefix of its first 8192, so an
    # archive built at seqlen 2048 would sail through the hash check on an 8192
    # loader — while covering only the first quarter of the window. That is not
    # misalignment but it is not the metric the name promises either.
    m_seq = manifest.get('seqlen')
    if m_seq and n_tokens and n_tokens > int(m_seq):
        raise ValueError(
            f"[key_token{where}] this loader yields {n_tokens} tokens per sample "
            f"but the archive was generated at seqlen={m_seq}: its intervals "
            f"cover only the first {m_seq} tokens, so the rest of the window "
            f"would silently carry no key tokens. Regenerate at seqlen="
            f"{n_tokens} or load the matching archive.")
    # The LongPPL knobs decide WHICH tokens are key, and nothing downstream can
    # tell a 512/128 archive from a 256/64 one -- the text hashes match either
    # way, so a protocol change would silently keep scoring the old positions.
    if knobs:
        diff = {k: (manifest.get(k), v) for k, v in knobs.items()
                if v is not None and manifest.get(k) is not None
                and int(manifest[k]) != int(v)}
        if diff:
            raise ValueError(
                f"[key_token{where}] the archive was built with "
                + ', '.join(f'{k}={a}' for k, (a, _) in diff.items())
                + " but this metric asks for "
                + ', '.join(f'{k}={b}' for k, (_, b) in diff.items())
                + " — those decide which tokens are key, and the text hashes "
                  "match either way, so nothing else would catch it. Regenerate "
                  "the archive with the metric's protocol.")
    m_n = manifest.get('n_slices')
    if m_n and n_slices and int(m_n) != int(n_slices):
        raise ValueError(
            f"[key_token{where}] the loader has {n_slices} samples but the "
            f"archive holds {m_n} (n_sample={manifest.get('n_sample')}): slice_i "
            f"would not refer to sample i.")
    recs = manifest.get('slices') or []
    if slice_idx >= len(recs):
        raise ValueError(
            f"[key_token{where}] the archive has {len(recs)} slices in its "
            f"manifest but slice_{slice_idx} was requested — this archive was "
            f"built for n_sample={manifest.get('n_sample')} "
            f"seqlen={manifest.get('seqlen')}.")
    rec = recs[slice_idx]
    if _text_sha(text) == rec['sha256']:
        return 'exact'
    if len(text) >= rec['chars'] and _text_sha(text[:rec['chars']]) == rec['sha256']:
        return 'prefix'
    raise ValueError(
        f"[key_token{where}] slice_{slice_idx} was computed on a DIFFERENT text "
        f"({rec['chars']} chars) than the loader produced ({len(text)} chars). "
        f"The intervals are character offsets into that text, so they would "
        f"mask the wrong tokens. Archive protocol: "
        f"{ {k: manifest.get(k) for k in ('dataset', 'n_sample', 'seqlen', 'min_seqlen', 'seed', 'target_model')} }. "
        f"Regenerate with gen_key_token.py using the same loader settings, or "
        f"point --key_token_path at the matching archive.")


def _loader_offsets(text, tokenizer, input_ids, skip_special_tokens=True):
    """Character spans of the LOADER's tokens, indexed by LOADER position.

    NOT `tokenizer(text, return_offsets_mapping=True)`: re-tokenizing the
    decoded text does not reproduce the loader's segmentation. decode() cannot
    restore a space that a token absorbed (" ." -> "."), so re-tokenizing
    re-merges and the indices drift — MEASURED: wikitext2 2048 -> 2032 tokens,
    qmsum 8192 -> 8183, and even gov_report drifts on some documents
    (2048 -> 2047 from position 241). cal_overlap returns indices into whatever
    sequence these offsets came from, and the loss mask is applied to the
    LOADER's ids, so any drift masks the wrong positions while still producing a
    plausible number.

    compute_offsets walks the loader's ids with incremental decoding instead, so
    len(offsets) == len(ids) by construction and the offsets track decode(ids)
    exactly (verified on gov_report / wikitext2: last offset == len(text)).
    ~0.1 s per 2048-token document.
    """
    offsets = compute_offsets(text, tokenizer, input_ids,
                              skip_special_tokens=skip_special_tokens)[0]
    if len(offsets) != input_ids.shape[-1]:
        raise ValueError(
            f"[key_token] computed {len(offsets)} offsets for "
            f"{input_ids.shape[-1]} loader tokens — the mask would be misaligned.")
    # The spans must tile the text exactly. Counting one token's characters
    # wrong shifts every later span, and cal_overlap would still return a
    # plausible-looking set of indices — the multi-byte failure above went
    # unnoticed precisely because nothing checked this.
    if len(offsets) and int(offsets[-1][1]) != len(text):
        raise ValueError(
            f"[key_token] offsets end at {int(offsets[-1][1])} but the text is "
            f"{len(text)} characters — the spans do not tile the document, so "
            f"the character intervals would map to the wrong tokens.")
    return offsets


def _retok_shift(loader_ids, retok_ids, tokenizer, where=''):
    """Index shift between the LOADER's token ids and the re-tokenized text.

    NO LONGER ON THE PIPELINE PATH — _loader_offsets removed the re-tokenization
    entirely. Kept as the diagnostic that MEASURES the drift (tests and the
    corpus audit use it to show which corpora would have been misaligned).

    Key-token intervals are CHARACTER spans, so they transfer across tokenizers
    (that is LongPPL's design). But cal_overlap returns indices into the
    RE-TOKENIZED sequence, while the loss mask is applied to the LOADER's ids.
    The original LongPPL never hits this: it feeds the model the very ids it
    re-tokenized. We do not — so the two sequences must be checked.

    Returns the offset to add to cal_overlap's indices. VERIFIED cases:
      gov_report  loader ids == retokenized ids (loader uses
                  add_special_tokens=False)                        -> 0
      wikitext2   loader prepends BOS and the round trip drifts by
                  16 tokens                                        -> raises
    Silently misaligned key tokens would mask the WRONG positions and still
    produce a plausible number, so an unmatched round trip is a hard error.
    """
    li = [int(x) for x in loader_ids]
    ri = [int(x) for x in retok_ids]
    if li == ri:
        return 0
    bos = getattr(tokenizer, 'bos_token_id', None)
    if bos is not None and li[:1] == [bos] and li[1:] == ri:
        return 1                     # loader kept BOS, the decoded text dropped it
    raise ValueError(
        f"[key_token{where}] the decoded text does not re-tokenize back to the "
        f"loader's ids (loader {len(li)} tok, re-tokenized {len(ri)} tok, first "
        f"mismatch at "
        f"{next((i for i in range(min(len(li), len(ri))) if li[i] != ri[i]), 'len')}"
        f"). Key-token indices are computed on the re-tokenized text, so they "
        f"would mask the wrong positions. Key tokens are only validated for "
        f"loaders that tokenize with add_special_tokens=False (gov_report); "
        f"for wikitext2/c4 the round trip is lossy.")


def merge_intervals(intervals):
    if intervals.size(0) == 0:
        return intervals

    start = intervals[:, 0]
    end = intervals[:, 1]
    adjacent = (start[1:] - end[:-1]) == 0

    keep_start_mask = torch.cat([torch.tensor([True]), ~adjacent])
    merged_start = start[keep_start_mask]
    keep_end_mask = torch.cat([~adjacent, torch.tensor([True])])
    merged_end = end[keep_end_mask]

    merged_intervals = torch.stack([merged_start, merged_end], dim=1)
    
    return merged_intervals 

def find_key_token(text, evaluator_model, evaluator_tokenizer, trunc_len, sliding_window, save_path='', alpha=2, beta=-2):
    text_encoded = evaluator_tokenizer(text, return_tensors="pt", return_offsets_mapping=True)               
    input_ids = text_encoded['input_ids'].to(evaluator_model.device)
    
    with torch.no_grad():
        output_full = evaluator_model(input_ids)
    shift_full_logits = output_full.logits
    # shift_full_logits = output_full.logits[:, :-1, :].contiguous()
    # shift_full_logits = shift_full_logits.reshape(-1, shift_full_logits.size(-1))
    
    loss_f = torch.nn.CrossEntropyLoss(reduction='none')
    bs, max_len = input_ids.shape
    key_tokens = []

    with torch.no_grad():
        for i, start_token in enumerate(range(0, max_len-trunc_len, sliding_window)):
            if start_token+trunc_len+sliding_window > max_len:
                sliding_window = max_len-start_token-trunc_len

            input_ids_short = input_ids[:, start_token: start_token+trunc_len+sliding_window]
            output_short = evaluator_model(input_ids_short)
            shift_short_logits = output_short.logits[:, trunc_len-1: trunc_len+sliding_window-1, :].contiguous()
            shift_short_logits = shift_short_logits.reshape(-1, shift_short_logits.size(-1))
            shift_short_labels = input_ids_short[:, trunc_len: trunc_len+sliding_window].reshape(-1)
            
            shift_full_trunc_logits = shift_full_logits[:, start_token+trunc_len-1: start_token+trunc_len+sliding_window-1, :].reshape(-1, shift_full_logits.size(-1))
            shift_full_labels = input_ids[:, start_token+trunc_len: start_token+trunc_len+sliding_window].reshape(-1)

            loss_full = loss_f(shift_full_trunc_logits, shift_full_labels)
            loss_short = loss_f(shift_short_logits, shift_short_labels)

            # loss_full = loss_f(output_full.logits[0, start_token+trunc_len-1: start_token+trunc_len+sliding_window-1, :], input_ids[0, start_token+trunc_len: start_token+trunc_len+sliding_window])
            # loss_short = loss_f(output_short.logits[0, trunc_len-1: trunc_len+sliding_window-1, :], input_ids_short[0, trunc_len: trunc_len+sliding_window])

            # loss_discrepancy = (torch.logical_and((loss_short - loss_full) > alpha, loss_full < (beta * -1))).squeeze()
            loss_discrepancy = (torch.logical_and((loss_short - loss_full) > alpha, loss_full < (beta * -1))).flatten()

            for i, is_key in enumerate(loss_discrepancy):
                if is_key:
                    key_tokens.append(start_token+trunc_len+i)
    
    # key_text_intervals = merge_intervals(text_encoded['offset_mapping'][0, key_tokens])
    # key_text_intervals = merge_intervals(text_encoded['offset_mapping'].reshape(-1, 2)[key_tokens])
    key_text_intervals = merge_intervals(text_encoded['offset_mapping'].squeeze(0)[key_tokens])

    if save_path:
        # tmp + rename: a slice file must never be half-written. A truncated
        # file parses as a SHORTER interval list, which is indistinguishable
        # from a document that genuinely had fewer key tokens -- and --resume
        # would then trust it.
        slices_str = ";".join([f"[{element[0]}, {element[1]}]" for element in key_text_intervals])
        tmp_path = save_path + '.tmp'
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(slices_str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, save_path)

    return key_text_intervals

def load_key_token(save_path):
    """Parse one archived slice file into a list of [start, end] intervals.

    Uses ast.literal_eval rather than eval(): these files are data, and eval()
    on file content is an arbitrary-code-execution path. literal_eval accepts
    exactly the list-of-ints we write and rejects everything else.
    Returns None for an empty file (a document with no key token).
    """
    with open(save_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue
            key_text_slices = []
            for key_slice in line.split(';'):
                key_slice = key_slice.strip()
                if not key_slice:
                    continue
                try:
                    interval = ast.literal_eval(key_slice)
                except (ValueError, SyntaxError) as e:
                    raise ValueError(
                        f"corrupt key-token file {save_path}: cannot parse "
                        f"{key_slice!r} ({e})")
                key_text_slices.append(interval)
            return key_text_slices

def cal_overlap(offset_mapping, key_text_slices):
    if key_text_slices is None:
        return None

    key_tokens = []
    i, j = 0, 0
    
    while i < len(offset_mapping) and j < len(key_text_slices):
        a_start, a_end = offset_mapping[i]
        b_start, b_end = key_text_slices[j]

        if a_start >= b_start and a_end <= b_end:
            # i-1: the key token is PREDICTED at shift position i-1. Token 0 has
            # no predecessor, so there is no position that predicts it -- and
            # -1 would quietly mark the LAST position of the sequence instead.
            if i - 1 >= 0:
                key_tokens.append(i-1)
            i += 1
        elif a_start < b_start:
            i += 1
        else:
            j += 1

    return key_tokens

_OFFSET_CTX = 256      # tokens of context when measuring a fragment run


def compute_offsets(text, tokenizer, input_ids, skip_special_tokens=True):
    """
    Compute character-level offset mappings for tokens when tokenizer doesn't support return_offsets_mapping.

    Parameters:
        text: Original text string
        tokenizer: Tokenizer instance
        input_ids: Tensor of shape [batch_size, seq_len] or [1, seq_len]

    Returns:
        Tensor of shape [batch_size, seq_len, 2] with [start, end] offsets for each token

    Multi-byte characters (CJK, emoji, accented text in some vocabularies) are
    split across SEVERAL byte-level BPE tokens, and decoding one such token on
    its own yields U+FFFD -- one character where the real text has none yet.
    Measuring each token in isolation therefore mis-counts the group and every
    later offset in the document is shifted. MEASURED on wikitext2 with gpt2:
    a document containing "杜甫" ended at offset 2196 for a 2195-character text,
    and 217/512 spans no longer matched their token. The upstream LongPPL
    implementation has the same behaviour, so this is a fix, not a port
    difference -- and it is a no-op where nothing is fragmented (verified
    identical on the gov_report archives: same 1729 key tokens, 0/8 documents
    changed).

    Fragmented runs are merged until they decode cleanly; the characters go to
    the LAST token of the run and the fragments get zero-width spans at the same
    position, so len(offsets) == len(input_ids) still holds and cal_overlap
    selects the token that completes the character.
    """
    batch_size, seq_len = input_ids.shape
    total_offsets = []
    # must match how `text` was decoded, or the spans stop tiling it
    _sst = bool(skip_special_tokens)

    for batch_idx in range(batch_size):
        ids = input_ids[batch_idx].tolist()
        offsets = []
        text_pointer = 0
        token_idx = 0

        while token_idx < seq_len:
            if token_idx == 0:
                # First token: decode it to get its length
                piece = tokenizer.decode([ids[0]], skip_special_tokens=_sst)
            else:
                # Subsequent tokens: decode cumulative to get incremental length
                prev_text = tokenizer.decode([ids[token_idx - 1]], skip_special_tokens=_sst)
                cumulative_text = tokenizer.decode(ids[token_idx - 1:token_idx + 1],
                                                   skip_special_tokens=_sst)
                piece = cumulative_text[len(prev_text):]

            if '\ufffd' not in piece:
                offsets.append([text_pointer, text_pointer + len(piece)])
                text_pointer += len(piece)
                token_idx += 1
                continue

            # piece is part of a character: extend until the run decodes cleanly
            last, group = token_idx, [ids[token_idx]]
            while last + 1 < seq_len:
                last += 1
                group.append(ids[last])
                if '\ufffd' not in tokenizer.decode(group, skip_special_tokens=_sst):
                    break
            # How many characters does the run actually add? Decoding the run
            # with one token of context is not enough: Mistral's byte-fallback
            # tokens decode to U+FFFD in ANY short context, and the group then
            # measures 2 characters too long (MEASURED on wikitext2: offsets
            # ended at 7582 for a 7580-character document). The prefix decode is
            # the ground truth, and it is only needed once per run.
            # Measured from a window that starts well before the run and is
            # decoded both with and without it, so the two decodes share the
            # same context and the difference is exactly what the run adds.
            # A full-prefix decode is equally correct but quadratic: 7.4 s for a
            # 4096-token CJK document vs 0.4 s here.
            win = max(0, token_idx - _OFFSET_CTX)
            piece_len = (len(tokenizer.decode(ids[win:last + 1], skip_special_tokens=_sst))
                         - len(tokenizer.decode(ids[win:token_idx], skip_special_tokens=_sst)))
            if piece_len < 0:
                piece_len = 0
            piece = ' ' * piece_len          # only its length is used below
            # Match the fast tokenizer's own convention (verified against
            # return_offsets_mapping on gpt2 and Llama-3.1 for ' 杜甫'): the
            # FIRST token of the run spans everything it introduced (including a
            # leading space), and each following fragment spans the character it
            # completes. Giving the fragments a zero-width span instead loses key
            # tokens, because the archive's intervals are HF-convention spans and
            # cal_overlap needs full containment.
            end = text_pointer + len(piece)
            offsets.append([text_pointer, end])
            for _ in range(len(group) - 1):
                offsets.append([max(end - 1, text_pointer), end])
            text_pointer = end
            token_idx = last + 1

        total_offsets.append(offsets)

    # Return as tensor: [batch_size, seq_len, 2]
    return torch.tensor(total_offsets)


# def compute_longppl(
#         text,
#         model,
#         evaluator_model=None,
#         tokenizer=None, 
#         evaluator_tokenizer=None, 
#         save_path='', 
#         load_path='',
#         key_token_list=None,
#         loss_func='longppl',  # 'longppl' or 'longjsd'
#         dense_logits_list=None,  # Required for 'longjsd'
#         trunc_len=4096, 
#         sliding_window=1024,
#         alpha=2,
#         beta=-2
#     ):
#     r"""
#     Compute the LongPPL or LongJSD for long text sequences.

#     Parameters:
#         text (`str` or `list`): 
#             The input text(s) for which LongPPL/LongJSD is calculated.
#         model (`transformers.PretrainedModel`): 
#             The model used for LongPPL/LongJSD calculation.
#         evaluator_model (`transformers.PretrainedModel`, *optional*): 
#             The evaluator model used to identify the key tokens (for online mode).
#         tokenizer (`transformers.PretrainedTokenizer`, *optional*): 
#             Tokenizer of the evaluated model.
#         evaluator_tokenizer (`transformers.PretrainedTokenizer`, *optional*): 
#             Tokenizer of the evaluator model (for online mode).
#         save_path (`str`, *optional*): If specified, the path to save the computed key tokens.
#         load_path (`str`, *optional*): If specified, the path to load precomputed key tokens (for offline mode).
#         key_token_list (`list`, *optional*): Pre-computed key token indices list. If provided, this takes priority.
#         loss_func (`str`, *optional*, default='longppl`): 'longppl' for standard LongPPL, 'longjsd' for JSD version.
#         dense_logits_list (`list` or `torch.Tensor`, *optional*): Dense logits for JSD calculation (required for 'longjsd').
#         trunc_len (`int`, *optional*, default=4096): Length of the truncated short context.
#         sliding_window (`int`, *optional*, default=1024): Number of tokens sharing the same short context.
#         alpha (`float`, *optional*, default=2): Threshold for LSD in key token detection.
#         beta (`float`, *optional*, default=-2): Threshold for LCL in key token detection.

#     Returns:
#         [`Dict`]: A `Dict` object including:
#             - 'longppl' (`float`, *optional*): The LongPPL score (for 'longppl' mode).
#             - 'longjsd' (`float`, *optional*): The LongJSD score (for 'longjsd' mode).
#             - 'n_key_token' (`int`): The number of key tokens (under the evaluated model).
#             - 'ppl' (`float`): The PPL score.
#             - 'n_token' (`int`): The number of tokens in the input text.
#     """
#     if loss_func == 'longjsd' and dense_logits_list is None:
#         raise ValueError("dense_logits_list must be provided for longjsd")
    
#     assert type(text) in [str, list]
#     if type(text) == str:
#         text = [text]
#     total_seqlen = 0
#     total_key_token_len = 0
#     nll_all_list = []
#     nll_key_list = []
#     jsd_key_list = []  # Store JSD values for each sequence
#     jsd_key_token_counts = []  # Store key token counts for each sequence (for weighted average)
    
#     for text_idx, cur_text in enumerate[str](text):
#         try:
#             encoded_input = tokenizer(cur_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
#             input_ids = encoded_input['input_ids'].to(model.device)
#             offset_mapping = encoded_input['offset_mapping'][0]
#         except NotImplementedError:
#             encoded_input = tokenizer(cur_text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=False)
#             input_ids = encoded_input['input_ids'].to(model.device)
#             offset_mapping = compute_offsets(cur_text, tokenizer, input_ids)[0]
        
#         # Get key tokens with priority: key_token_list > load_path > evaluator_model
#         key_tokens = None
#         key_text_slices = None
        
#         if key_token_list is not None and text_idx < len(key_token_list):
#             # Use precomputed key token indices
#             key_tokens = key_token_list[text_idx]
#             if not isinstance(key_tokens, list):
#                 if isinstance(key_tokens, torch.Tensor):
#                     key_tokens = key_tokens.cpu().tolist()
#                 else:
#                     key_tokens = [key_tokens]
#         elif load_path and os.path.exists(load_path):
#             # Load from file (offline mode)
#             key_text_slices = load_key_token(load_path)
#             if key_text_slices is not None:
#                 key_tokens = cal_overlap(offset_mapping, key_text_slices)
#         elif evaluator_model is not None:
#             # Compute key tokens (online mode)
#             key_text_slices = find_key_token(cur_text, evaluator_model, evaluator_tokenizer, trunc_len, sliding_window, save_path, alpha, beta)
#             if key_text_slices is not None:
#                 key_tokens = cal_overlap(offset_mapping, key_text_slices)
        
#         bs, seqlen = input_ids.shape
#         key_token_len = len(key_tokens) if key_tokens is not None else 0
        
#         with torch.no_grad():
#             outputs = model(input_ids)
#         lm_logits = outputs.logits
            
#         shift_logits = lm_logits[:, :-1, :].contiguous()
#         shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
#         shift_labels = input_ids[:, 1:].reshape(-1)
        
#         loss_func_ce = torch.nn.CrossEntropyLoss()
#         loss_all = loss_func_ce(shift_logits, shift_labels)
        
#         nll_all = loss_all.float() * seqlen * bs
#         nll_all_list.append(nll_all)
#         total_seqlen += seqlen * bs
        
#         if key_tokens is not None and len(key_tokens) > 0:
#             # Filter key_tokens to valid indices
#             valid_key_tokens = [kt for kt in key_tokens if 0 <= kt < shift_logits.shape[0]]
            
#             if len(valid_key_tokens) > 0:
#                 valid_key_tokens_tensor = torch.tensor(valid_key_tokens, device=shift_logits.device)
                
#                 if loss_func == 'longjsd':
#                     # Compute JSD on key tokens
#                     dense_logits_seq = None
#                     if dense_logits_list is not None:
#                         if isinstance(dense_logits_list, list) and text_idx < len(dense_logits_list):
#                             dense_logits_seq = dense_logits_list[text_idx]
#                         elif isinstance(dense_logits_list, torch.Tensor):
#                             # Handle different tensor shapes: [batch, seq, vocab] or [seq, vocab]
#                             if len(dense_logits_list.shape) == 3:
#                                 # [batch, seq, vocab]
#                                 if dense_logits_list.shape[0] > text_idx:
#                                     dense_logits_seq = dense_logits_list[text_idx]
#                             elif len(dense_logits_list.shape) == 2:
#                                 # [seq, vocab] - single sequence
#                                 if len(text) == 1:
#                                     dense_logits_seq = dense_logits_list
                    
#                     if dense_logits_seq is not None:
#                         # Handle shape: [seq_len, vocab_size] or [batch, seq_len, vocab_size]
#                         if len(dense_logits_seq.shape) == 3:
#                             dense_logits_seq = dense_logits_seq[0]
#                         # Ensure device match
#                         if isinstance(dense_logits_seq, torch.Tensor):
#                             dense_logits_seq = dense_logits_seq.to(model.device)
#                         # Shift to match shift_logits
#                         dense_logits_seq = dense_logits_seq[:-1, :].contiguous()
#                         dense_logits_seq = dense_logits_seq.reshape(-1, dense_logits_seq.size(-1))
                        
#                         # Compute JSD on key tokens
#                         jsd_loss = JSD()
#                         jsd_key = jsd_loss(
#                             shift_logits[valid_key_tokens_tensor], 
#                             dense_logits_seq[valid_key_tokens_tensor]
#                         )
#                         # Store JSD value and token count for weighted average
#                         jsd_key_list.append(jsd_key.item())
#                         jsd_key_token_counts.append(len(valid_key_tokens) * bs)
#                         total_key_token_len += len(valid_key_tokens) * bs
#                     else:
#                         # Fallback to standard cross-entropy on key tokens if no dense logits
#                         loss_key = loss_func_ce(shift_logits[valid_key_tokens_tensor], shift_labels[valid_key_tokens_tensor])
#                         nll_key = loss_key.float() * len(valid_key_tokens) * bs
#                         nll_key_list.append(nll_key)
#                         total_key_token_len += len(valid_key_tokens) * bs
#                 else:
#                     # Standard LongPPL: use cross-entropy loss
#                     loss_key = loss_func_ce(shift_logits[valid_key_tokens_tensor], shift_labels[valid_key_tokens_tensor])
#                     nll_key = loss_key.float() * len(valid_key_tokens) * bs
#                     nll_key_list.append(nll_key)
#                     total_key_token_len += len(valid_key_tokens) * bs
    
#     ppl_all = torch.exp(sum(nll_all_list) / total_seqlen) if total_seqlen > 0 else None
    
#     result = {
#         "n_key_token": total_key_token_len,
#         "ppl": ppl_all,
#         "n_token": total_seqlen
#     }
    
#     if loss_func == 'longppl':
#         ppl_key = torch.exp(sum(nll_key_list) / total_key_token_len) if total_key_token_len > 0 else None
#         result["longppl"] = ppl_key
#     elif loss_func == 'longjsd':
#         if len(jsd_key_list) > 0 and len(jsd_key_token_counts) > 0:
#             # Weighted average of log JSD values
#             log_jsds = np.log(np.array(jsd_key_list))
#             weights_key = np.array(jsd_key_token_counts)
#             longjsd = np.exp((log_jsds * weights_key).sum() / weights_key.sum())
#             result["longjsd"] = longjsd
#         else:
#             # Fallback to cross-entropy if no JSD computed
#             if len(nll_key_list) > 0 and total_key_token_len > 0:
#                 ppl_key = torch.exp(sum(nll_key_list) / total_key_token_len)
#                 result["longjsd"] = ppl_key.item()
#             else:
#                 result["longjsd"] = None
    
#     return result

    
        
        # loss_f = torch.nn.CrossEntropyLoss(reduction='none')
        # loss_overall = loss_f(output_full.logits[0, :-1, :], input_ids[0, 1:]).to(torch.float).cpu().numpy()
        
        # if key_tokens is None or len(key_tokens) == 0:
        #     print("No key tokens!")
        #     return {"longppl": None, "n_key_token": None, "ppl": np.exp(loss_overall.mean()), "n_token": input_ids.shape[-1]}

        # loss_key = loss_overall[key_tokens]

        # return {"longppl": np.exp(loss_key.mean()), "n_key_token": len(key_tokens), "ppl": np.exp(loss_overall.mean()), "n_token": input_ids.shape[-1]}