import torch
from .awq import AWQ
from .gptq import GPTQ
from .qeft import QEFT
import gc

from accelerate import dispatch_model

METHOD = {
    'gptq': GPTQ,
    'awq': AWQ,
    'qeft': QEFT
}

def _make_arch_key(arch):
    """Create a hashable string key from a W-arch dict for AWQ result caching."""
    import json
    def _to_list(v):
        if hasattr(v, 'tolist'):
            return v.tolist()
        return [int(x) for x in v]
    return json.dumps({k: _to_list(v) for k, v in arch.items()}, sort_keys=True)


def get_quantized_model(method, arch, model_name, device_map, group_size=128, dtype='auto', config=None, dev='cuda', prune=False, do_owq=False, owq_path=None, awq_results_cache=None, **kwargs):
    method_obj = METHOD[method](model_name=model_name, config=config, device_map=device_map, group_size=group_size, dtype=dtype, dev=dev, arch=arch, prune=prune, do_owq=do_owq, owq=owq_path, **kwargs)

    if prune:
        print('Pruning the model')
        method_obj.prune_model()

    if method == 'awq' and awq_results_cache is not None:
        arch_key = _make_arch_key(arch)
        cached = awq_results_cache.get(arch_key)
        awq_results = method_obj.run(cached_awq_results=cached)
        if cached is None:
            awq_results_cache[arch_key] = awq_results
            print(f"[AWQ cache] Stored calibration results (cache size: {len(awq_results_cache)})")
    else:
        method_obj.run()

    model = dispatch_model(method_obj.model, method_obj.device_map)
    del method_obj
    torch.cuda.empty_cache()
    gc.collect()

    return model
