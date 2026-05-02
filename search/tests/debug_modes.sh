#!/bin/bash
# Run scripts/post_search_split.sh with one of three sample-mode overrides
# without modifying the original. The original is copied to /tmp and the
# --random_sample / --quantile_sample lines are toggled there.
#
# Usage (from /NAS/SJ/actquant/search):
#   bash tests/debug_modes.sh A   # random only
#   bash tests/debug_modes.sh B   # quantile only
#   bash tests/debug_modes.sh C   # quantile + random
set -e
MODE=${1:-C}
ORIG=scripts/post_search_split.sh
TMP=/tmp/post_search_split_${MODE}.sh

python3 - "$ORIG" "$TMP" "$MODE" <<'PY'
import sys, pathlib
src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]
text = pathlib.Path(src).read_text()
RAND = '--random_sample ${RANDOM_SAMPLE} \\\n'
QUANT = '--quantile_sample ${QUANTILE_SAMPLE}"'
RAND_END = '--random_sample ${RANDOM_SAMPLE}"'   # trailing close quote variant
if mode == 'A':
    # Drop --quantile_sample, keep --random_sample (ending the ARGS string).
    text = text.replace(RAND + QUANT, RAND_END)
elif mode == 'B':
    # Drop --random_sample, keep --quantile_sample.
    text = text.replace(RAND, '')
elif mode == 'C':
    pass  # original (quantile + random)
else:
    sys.exit(f'Unknown mode {mode!r}')
pathlib.Path(dst).write_text(text)
PY

chmod +x "$TMP"
echo "=== MODE $MODE — relevant flags ==="
grep -E "^--random_sample|^--quantile_sample" "$TMP" | head
echo "==================================="
exec timeout 240 bash "$TMP" 0
