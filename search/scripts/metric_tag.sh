# metric_tag.sh — SAVE 디렉터리에 붙일 짧은 지표 태그. 다른 스크립트에서 source.
#
#   metric_tag_from_tasks "gov_jsd_pp512_s128 wt2_ppl"   -> _mgovj+1
#   metric_tag_from_knobs  wikitext2 jsd loss            -> _mwt2j
#
# dir 이름에는 이미 _st128_pp512 / 샘플수 같은 파라미터가 들어가고 255자 한계에
# 가까우므로 태그는 dataset+목적함수 4~6자만. utils/metric_specs.py의 이름 규칙
# (<dataset>_<loss>_<...>)을 그대로 따르므로 두 표기가 서로 대응된다.

_ds_code() {   # dataset 이름 -> 3자 코드
    case "$1" in
        wikitext2*)  echo wt2 ;;
        gov_report*) echo gov ;;
        c4*)         echo c4  ;;
        gsm8k*)      echo g8k ;;
        needle*)     echo ndl ;;
        *)           echo "${1:0:3}" ;;
    esac
}

# 이름 목록 -> 첫 태스크 코드 + 나머지 개수. 정확한 목록은 results.csv에 남는다.
metric_tag_from_tasks() {
    local first ds loss n tag
    first=${1%% *}                       # gov_jsd_pp512_s128
    [ -z "${first}" ] && return
    ds=$(_ds_code "${first%%_*}")        # gov
    loss=${first#*_}; loss=${loss%%_*}   # jsd | ppl
    n=$(echo $1 | wc -w)
    tag="_m${ds}${loss:0:1}"
    [ ${n} -gt 1 ] && tag="${tag}+$((n-1))"
    echo "${tag}"
}

# knob에서 직접:
#   $1=dataset $2=loss_func $3=metric(loss|ppl) $4=n_sample $5=seqlen $6=min_seqlen
# -> _mwt2j_n128q2048        (min_seqlen=0이면 생략)
# -> _mgovj_n8q8196m8192
# n_sample/seqlen/min_seqlen까지 넣는 이유: stride(_st)·답변창(_pp)·residual(_r)·
# sink(_sk)는 dir에 이미 있지만 이 셋은 어느 스크립트에도 없어서, N_SAMPLE만 바꾼
# 런이 타임스탬프 빼고 같은 이름이 된다. $4~$6은 생략 가능(옛 호출 호환).
metric_tag_from_knobs() {
    local ds tag
    ds=$(_ds_code "${1%% *}")
    if [ "${3:-loss}" == "ppl" ]; then tag="_m${ds}p"; else tag="_m${ds}${2:0:1}"; fi
    [ -n "$4" ] && tag="${tag}_n$4"
    [ -n "$5" ] && tag="${tag}q$5"
    [ -n "$6" ] && [ "$6" != "0" ] && tag="${tag}m$6"
    echo "${tag}"
}
