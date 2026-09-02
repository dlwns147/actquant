# metric_task.sh — resolve a NAMED calibration metric (utils/metric_specs.py, the same
# names correlation.py --metrics and post_search.py --metric_tasks use) into THIS
# script's knobs. Sourced by search.sh / second_search.sh / second_search_new.sh.
#
#   metric_task_apply <name> <model_name> [key_token_base] [--loss_only]
#   metric_task_stamp <save_dir>
#
# WHERE THE RESOLUTION LIVES — here, and only here. search.py / second_search*.py take
# raw knobs and know nothing about metric names: a name is a WRAPPER-level convenience,
# so the registry lookup happens once, in the shell, before the arg list and the SAVE-dir
# tags are built from the resolved values. That keeps the measurement and the directory
# name derived from the SAME resolution instead of from two copies of it, and it keeps
# the python entry points unchanged.
#
# metric_task_apply overwrites DATASET / USE_CHAT_TEMPLATE / N_SAMPLE / SEQLEN /
# MIN_SEQLEN / METRIC / LOSS_FUNC / STRIDE / PREFILL_PROMPT / LAST_TOKENS / SCORE /
# USE_KEY_TOKEN / TRUNC_LEN / SLIDING_WINDOW / ALPHA / BETA / KEY_TOKEN_PATH /
# METRIC_SPEC, and PRINTS every knob whose value it CHANGED.
#
# IF A KNOB IS SET BY HAND *AND* METRIC_TASK IS SET, THE NAME WINS — visibly. A metric
# name IS a protocol, so there is nothing for a hand-set DATASET to add; what would be
# wrong is discarding it silently, so every override is printed as `KNOB: old -> new`.
# To keep a hand-set knob, clear METRIC_TASK: the two are alternatives, not layers.
#
# `--loss_only` refuses metrics the 2nd-stage entry points cannot measure (PPL, key
# token): they call evaluator.eval(..., 'loss') and neither they nor utils/awq_pool's
# worker is handed a key-token archive.
# KEY_TOKEN_PATH comes back as the DERIVED kt_eval-<evaluator>_tgt-<model> root — the
# value --key_token_path takes — and the archive is verified before any GPU work.
metric_task_apply() {
    local name=$1 model=$2 ktbase=${3:-key_token} extra=${4:-}
    [ -z "${name}" ] && return 0
    local knobs
    knobs=$(python -m utils.metric_specs --shell "${name}" --model_name "${model}" \
                   --key_token_path "${ktbase}" ${extra}) || exit 1
    # the key-token protocol is INERT unless the metric is key-token weighted, so a
    # change there is not a change to what gets measured -- don't report it as one
    local kt=0 skip=""
    case "${knobs}" in *"USE_KEY_TOKEN='True'"*) kt=1 ;; esac
    [ ${kt} -eq 0 ] && skip=" TRUNC_LEN SLIDING_WINDOW ALPHA BETA KEY_TOKEN_PATH "
    local k v changed=()
    while IFS='=' read -r k v; do
        [ -z "${k}" ] && continue
        v=${v#\'}; v=${v%\'}
        [ -n "${skip}" ] && [[ "${skip}" == *" ${k} "* ]] && continue
        if [ -n "${!k+x}" ] && [ "${!k}" != "${v}" ]; then
            changed+=("${k}: ${!k} -> ${v}")
        fi
    done <<< "${knobs}"
    eval "${knobs}"
    echo "[metric_task] ${name} (spec ${METRIC_SPEC})"
    if [ ${#changed[@]} -gt 0 ]; then
        echo "  the name overrides these knobs set in this script:"
        printf '    %s\n' "${changed[@]}"
    else
        echo "  every knob in this script already matched the name"
    fi
}

# Record WHICH named metric an archive optimised, next to the archive itself. The python
# side stores the resolved protocol (protocol_dict -> iter_<it>.stats) but not the name,
# because it never saw one; this is the shell's half of that provenance. spec is
# metric_specs.spec_sha8 — the hash of the metric's DEFINITION, so a later edit to the
# group/task is detectable rather than silently redefining an archive's objective.
metric_task_stamp() {
    local save=$1
    [ -z "${METRIC_TASK}" ] && return 0
    mkdir -p "${save}"
    printf '{"metric_task": "%s", "spec": "%s", "resolved_by": "scripts/metric_task.sh"}\n' \
        "${METRIC_TASK}" "${METRIC_SPEC}" > "${save}/metric_task.json"
}
