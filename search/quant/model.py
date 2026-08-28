import torch
from .awq import AWQ
from .gptq import GPTQ
from .qeft import QEFT
from .awq_qeft import AWQ_QEFT
import gc

from accelerate import dispatch_model

METHOD = {
    'gptq': GPTQ,
    'awq': AWQ,
    'qeft': QEFT,
    'awq_qeft': AWQ_QEFT,
}

# The calibration-set kwarg is spelled differently by each quantizer's run().
# None (the default) means "don't pass it" -> each method keeps its own default
# (AWQ pileval 128x512, GPTQ/QEFT c4 128x2048), so existing callers are byte-
# identical to before. awq_qeft.run() takes no calib argument and is omitted.
_CALIB_KW = {'awq': 'calib_data', 'gptq': 'calib', 'qeft': 'dataset'}


def get_quantized_model(method, arch, model_name, device_map, group_size=128, dtype='auto', config=None, dev='cuda', prune=False, do_owq=False, owq_path=None, w_calib=None, w_act_order=None, **kwargs):
    method_name = method
    method = METHOD[method](model_name=model_name, config=config, device_map=device_map, group_size=group_size, dtype=dtype, dev=dev, arch=arch, prune=prune, do_owq=do_owq, owq=owq_path, **kwargs)

    if prune:
        print('Pruning the model')
        method.prune_model()

    run_kw = {}
    if w_calib:
        kw = _CALIB_KW.get(method_name)
        if kw is None:
            raise ValueError(f"w_calib is not supported for method '{method_name}'")
        run_kw[kw] = w_calib
        print(f'[quant] {method_name}: calibration set = {w_calib}')
    if w_act_order is not None and method_name in ('gptq', 'qeft'):
        # 'auto' opts into each method's own model gate; True/False are honoured
        # verbatim. None (default) = leave run()'s own default alone.
        run_kw['act_order'] = w_act_order
        print(f'[quant] {method_name}: act_order = {w_act_order}')
    method.run(**run_kw)    
    model = dispatch_model(method.model, method.device_map)
    del method
    torch.cuda.empty_cache
    gc.collect()

    return model