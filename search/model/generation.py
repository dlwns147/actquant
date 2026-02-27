import warnings
from typing import Dict

import torch
from transformers.generation.configuration_utils import (
    GenerationConfig,
    NEED_SETUP_CACHE_CLASSES_MAPPING,
    QUANT_BACKEND_CLASSES_MAPPING,
)
from transformers.cache_utils import (
    DynamicCache,
    EncoderDecoderCache,
    OffloadedCache,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, is_hqq_available, is_optimum_quanto_available
from transformers.generation.utils import GenerationMixin
from transformers.generation.configuration_utils import CACHE_CONFIG_MAPPING, QUANT_BACKEND_CLASSES_MAPPING

from model.KIVICache import KIVICacheConfig, KIVIDynamicCache, KIVIFakeCache, ThinkKIVIDynamicCache, ThinkKIVIFakeCache
from model.HQQCache import QuantizedCacheConfig, HQQQuantizedCache

logger = logging.get_logger(__name__)

QUANT_BACKEND_CLASSES_MAPPING['HQQ'] = HQQQuantizedCache
CACHE_CONFIG_MAPPING['quantized'] = QuantizedCacheConfig


def convert_generation(model_config):
    def _prepare_cache_for_generation(
        self,
        generation_config: GenerationConfig,
        model_kwargs: Dict,
        assistant_model: "PreTrainedModel",
        batch_size: int,
        max_cache_length: int,
        device: torch.device,
    ) -> bool:
        """
        Prepares the cache for generation (if applicable), given `generate`'s parameterization. If a cache is
        instantiated, writes it to `model_kwargs`, under the name expected by the model.
        """

        cache_name = "past_key_values" if "mamba" not in self.__class__.__name__.lower() else "cache_params"
        requires_cross_attention_cache = (
            self.config.is_encoder_decoder or model_kwargs.get("encoder_outputs") is not None
        )

        # Quick escape route 1: if the user specifies a cache, we only need to:
        # a) check for conflicting `generate` arguments
        # b) convert to the new cache format (if the user passes a legacy cache and model supports it)
        user_defined_cache = model_kwargs.get(cache_name)
        if user_defined_cache is not None:
            if generation_config.cache_implementation is not None:
                raise ValueError(
                    f"Passing both `cache_implementation` (used to initialize certain caches) and `{cache_name}` (a "
                    "Cache object) is unsupported. Please use only one of the two."
                )
            if isinstance(user_defined_cache, tuple) and self._supports_default_dynamic_cache():
                model_kwargs[cache_name] = (
                    DynamicCache.from_legacy_cache(user_defined_cache)
                    if not requires_cross_attention_cache
                    else EncoderDecoderCache.from_legacy_cache(user_defined_cache)
                )
            return

        # Quick escape route 2: if the user specifies no cache is to be used. (conflicting arguments are handled in
        # `generation_config.validate()`)
        if generation_config.use_cache is False:
            return

        # Quick escape route 3: model that only supports legacy caches = nothing to prepare
        if not self._supports_default_dynamic_cache():
            if generation_config.cache_implementation is not None:
                warnings.warn(
                    "This model does not support `Cache` instances, it only supports the legacy cache format (tuple "
                    f"of tuples). `cache_implementation` (set to {generation_config.cache_implementation}) will be "
                    "ignored.",
                    UserWarning,
                )
            return

        # Otherwise we NEED to prepare a cache, based on `generation_config.cache_implementation`

        # TODO(joao): support static caches in assisted generation. assisted generation needs to roll back caches,
        # which is only supported in dynamic caches atm
        if assistant_model is not None and generation_config.cache_implementation is not None:
            logger.warning_once(
                "An assistant model is provided, using a dynamic cache instead of a cache of type="
                f"'{generation_config.cache_implementation}'."
            )
            generation_config.cache_implementation = None

        if generation_config.cache_implementation is not None:
            if generation_config.cache_implementation in NEED_SETUP_CACHE_CLASSES_MAPPING:
                if generation_config.cache_implementation == "static" and not self._supports_static_cache:
                    raise ValueError(
                        "This model does not support `cache_implementation='static'`. Please check the following "
                        "issue: https://github.com/huggingface/transformers/issues/28981"
                    )
                model_kwargs[cache_name] = self._get_cache(
                    cache_implementation=generation_config.cache_implementation,
                    batch_size=max(generation_config.num_beams, generation_config.num_return_sequences) * batch_size,
                    max_cache_len=max_cache_length,
                    device=device,
                    model_kwargs=model_kwargs,
                )
            elif generation_config.cache_implementation == "quantized":
                if not self._supports_quantized_cache:
                    raise ValueError(
                        "This model does not support the quantized cache. If you want your model to support quantized "
                        "cache, please open an issue and tag @zucchini-nlp."
                    )

                cache_config = (
                    generation_config.cache_config
                    if generation_config.cache_config is not None
                    else QuantizedCacheConfig()
                )
                cache_class = QUANT_BACKEND_CLASSES_MAPPING[cache_config.backend]

                if cache_config.backend == "quanto" and not is_optimum_quanto_available():
                    raise ImportError(
                        "You need to install optimum-quanto in order to use KV cache quantization with optimum-quanto backend. "
                        "Please install it via  with `pip install optimum-quanto`"
                    )
                elif cache_config.backend == "HQQ" and not is_hqq_available():
                    raise ImportError(
                        "You need to install `HQQ` in order to use KV cache quantization with HQQ backend. "
                        "Please install it via  with `pip install hqq`"
                    )

                model_kwargs[cache_name] = cache_class(cache_config)
            elif generation_config.cache_implementation == "offloaded":
                model_kwargs[cache_name] = OffloadedCache()
            elif generation_config.cache_implementation == "dynamic" \
                or generation_config.cache_implementation == "kivi":
                kv_method = getattr(self.config, "kv_method", [])
                use_think = "think" in kv_method
                if model_config.kivi_config.packing:
                    cache_cls = ThinkKIVIDynamicCache if use_think else KIVIDynamicCache
                else:
                    cache_cls = ThinkKIVIFakeCache if use_think else KIVIFakeCache
                model_kwargs[cache_name] = cache_cls(model_config.kivi_config)

        # Use DynamicCache() instance by default. This will avoid back and forth from legacy format that
        # keeps copying the cache thus using much more memory
        else:
            kv_method = getattr(self.config, "kv_method", [])
            use_think = "think" in kv_method
            print(f'kv_method: {kv_method}, use_think: {use_think}')
            if model_config.kivi_config.packing:
                cache_cls = ThinkKIVIDynamicCache if use_think else KIVIDynamicCache
            else:
                cache_cls = ThinkKIVIFakeCache if use_think else KIVIFakeCache
            model_kwargs[cache_name] = (
                cache_cls(model_config.kivi_config)
                if not requires_cross_attention_cache
                else EncoderDecoderCache(
                    cache_cls(model_config.kivi_config),
                    cache_cls(model_config.kivi_config),
                )
            )

    GenerationMixin._prepare_cache_for_generation = _prepare_cache_for_generation