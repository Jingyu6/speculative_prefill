""" Adopted from https://github.com/microsoft/MInference/blob/main/minference/patch.py """
import os


def minference_patch_vllm_tp(self, config_file, patch_config):
    self.model_runner.model.apply(
        minference_patch_vllm_executor(config_file, patch_config)
    )

def minference_patch_vllm_executor(config_file: str, patch_config={}):
    import json
    from collections import defaultdict

    import vllm
    from minference.modules.minference_forward import (
        gather_last_q_vertical_slash_topk_vllm, minference_vllm_forward)
    from vllm.attention import Attention
    from vllm.model_executor.models.chatglm import (GLMAttention, GLMBlock,
                                                    GLMTransformer)
    from vllm.model_executor.models.llama import (LlamaAttention,
                                                  LlamaDecoderLayer,
                                                  LlamaModel)

    vllm_version = vllm.__version__

    config = defaultdict(dict)
    if os.path.exists(config_file):
        config = json.load(open(config_file))
    attn_forward = minference_vllm_forward(
        config, vllm_version=vllm_version, patch_config=patch_config
    )
