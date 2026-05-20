"""
ifeval-pp (RebeccaYU920/ifeval-pp) task helpers.

ifeval-pp is a "paraphrased / perturbed" variant of Google's IFEval. The schema
matches IFEval except that:

  * `key` is a string (e.g. "1000:ct_alteration:1") instead of an int, and
  * every entry of `kwargs` carries the full set of argument keys with `null`
    placeholders for the unused ones.

It also **perturbs the constraint values**: alongside IFEval's supported
comparison relations (`'less than'`, `'at least'`) it introduces
`'at most'` (~42) and `'around'` (~17). The upstream IFEval instruction
checker hard-rejects those with

    ValueError: The supported relation for comparison must be in
                ('less than', 'at least'), but around is given.

which aborts the *entire* evaluation. We therefore reuse the upstream
instruction registry and the loose/strict response variants, but evaluate
each instruction **independently and defensively**: if the upstream checker
cannot construct or apply an instruction (perturbed/unsupported kwargs), that
single instruction is scored as *not followed* rather than crashing the run.
This is the standard IFEval convention ("a constraint that cannot be verified
is not satisfied") and is robust to any perturbation, not just these two
relations. We do NOT invent semantics for `'around'`/`'at most'`.
"""

from lm_eval.tasks.ifeval import instructions_registry
from lm_eval.tasks.ifeval.utils import InputExample, OutputExample, agg_inst_level_acc


__all__ = ["process_results", "agg_inst_level_acc"]


def _check_one(instruction_id, kwargs, prompt, responses):
    """True iff *any* response variant satisfies this single instruction.

    `responses` is one string (strict) or the list of loose variants. Any
    failure to build/apply the instruction (e.g. perturbed relation the
    upstream checker rejects) counts as "not followed".
    """
    try:
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        # Drop None/falsy kwargs, exactly like upstream, to avoid passing
        # null placeholders into build_description.
        clean = {k: v for k, v in kwargs.items() if v}
        instruction.build_description(**clean)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=prompt)

        for r in responses:
            if r.strip() and instruction.check_following(r):
                return True
        return False
    except Exception:  # noqa: BLE001 - perturbed/unsupported instruction => not followed
        return False


def _evaluate(inp, response_variants):
    is_following_list = [
        _check_one(iid, inp.kwargs[idx], inp.prompt, response_variants)
        for idx, iid in enumerate(inp.instruction_id_list)
    ]
    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response_variants[0],
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def _loose_variants(response):
    # Exactly the 8 transforms used by upstream test_instruction_following_loose.
    r = response.split("\n")
    remove_first = "\n".join(r[1:]).strip()
    remove_last = "\n".join(r[:-1]).strip()
    remove_both = "\n".join(r[1:-1]).strip()
    revised = response.replace("*", "")
    return [
        response,
        revised,
        remove_first,
        remove_last,
        remove_both,
        remove_first.replace("*", ""),
        remove_last.replace("*", ""),
        remove_both.replace("*", ""),
    ]


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],  # str in ifeval-pp; unused by the checker, kept for traceability
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]

    out_strict = _evaluate(inp, [response])
    out_loose = _evaluate(inp, _loose_variants(response))

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }
