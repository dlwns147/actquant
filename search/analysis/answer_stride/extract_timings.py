"""
Extract per-iteration wall-clock & forward-count estimates for each
(answer_token, stride) run, so we can quantify the speed/coverage trade-off.

Forward count per sample =
  1 (prefill of seq - last_tokens) + ceil(last_tokens / stride)
"""
import os
import json
import math
import glob
import csv

THINK_DIR = "/NAS/SJ/actquant/search/save/search/think"
OUT = "/NAS/SJ/actquant/search/analysis/answer_stride/data/timings.csv"

# (label, dir-prefix-or-glob, total_seq_used, ans, stride, n_sample)
RUNS = [
    ("pp128_s32",            "2605071157", 2048,  128,  32, 32),
    ("pp256_s64",            "2605070434", 2304,  256,  64, 64),
    ("pp256_s128",           "2605071142", 2048,  256, 128, 32),
    ("pp512_s128_n32",       "2605071156", 2048,  512, 128, 32),
    ("pp512_s128_n64",       "2605070643", 2560,  512, 128, 64),
    ("pp512_s128_seq2k",     "2605070828", 2048,  512, 128, 32),
    ("pp1024_s512",          "2605070803", 2048, 1024, 512, 64),
    # follow-up sweep 2026-05-07 19:29:
    ("pp512_s64_v2",         "2605071929_*1536seq_*64stride_pp512",  2048,  512,  64, 32),
    ("pp256_s128_resume_v2", "2605071929_*1792seq_*128stride_pp256", 2048,  256, 128, 32),
    ("pp1024_s128_v2_fixed", "2605080305_*2048seq_*128stride_pp1024",3072, 1024, 128, 32),
]

rows = []
for label, prefix, total_seq, ans, stride, nsamp in RUNS:
    pat = (os.path.join(THINK_DIR, prefix) if "*" in prefix
           else os.path.join(THINK_DIR, f"{prefix}_*kvdim*"))
    matches = glob.glob(pat)
    if not matches:
        print(f"[skip] {label}")
        continue
    run_dir = matches[0]
    iter_files = sorted(glob.glob(os.path.join(run_dir, "iter_*.stats")))
    times = []
    for f in iter_files:
        with open(f) as fh:
            d = json.load(fh)
        t = d.get("surrogate", {}).get("total_time", None)
        times.append(t)
    times = [t for t in times if t is not None]
    avg_time = sum(times) / len(times) if times else None
    n_chunks = math.ceil(ans / stride)
    fwd_per_sample = 1 + n_chunks
    fwd_per_eval = fwd_per_sample * nsamp  # one batch=1 eval per sample
    prefill = total_seq - ans
    # rough compute proxy: prefill_seq^2/2 + chunks * stride * (avg_kv_len)
    avg_kv_len = prefill + ans / 2
    compute = prefill * prefill / 2 + n_chunks * stride * avg_kv_len
    rows.append({
        "label": label,
        "ans": ans,
        "stride": stride,
        "n_sample": nsamp,
        "total_seq": total_seq,
        "prefill_seq": prefill,
        "n_chunks": n_chunks,
        "fwd_per_sample": fwd_per_sample,
        "fwd_per_eval": fwd_per_eval,
        "compute_proxy": compute,
        "avg_iter_time_s": avg_time,
        "iters_completed": len(times),
    })

print(f"{'label':<22} {'ans':>4} {'stride':>6} {'nsamp':>5} {'fwd/s':>6} "
      f"{'fwd/e':>6} {'iter_t(s)':>10}")
print("-" * 78)
for r in rows:
    print(f"{r['label']:<22} {r['ans']:>4} {r['stride']:>6} {r['n_sample']:>5} "
          f"{r['fwd_per_sample']:>6} {r['fwd_per_eval']:>6} "
          f"{(r['avg_iter_time_s'] or 0):>10.1f}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(f"[write] {OUT}")
