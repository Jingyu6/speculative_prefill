""" Adopted from https://github.com/microsoft/MInference/blob/main/minference/patch.py """
import os
from typing import List, Optional, Tuple

import torch


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

    def vllm_attn_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata,
        kv_scale: float = 1.0,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        # check self._kv_scale
        kv_scale = getattr(self, "_kv_scale", getattr(self, "_k_scale", kv_scale))
        return self.impl.forward(
            query, key, value, kv_cache, attn_metadata, kv_scale, layer_idx
        )

    def llama_model_forward_vllm(
        self,
        input_ids: Optional[torch.Tensor],
        positions: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)
        residual = None
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i],
                attn_metadata,
                residual,
                layer_idx=i,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def chatglm_model_forward_vllm(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_caches: List[torch.Tensor],
        attn_metadata,
    ) -> torch.Tensor:
        for i in range(self.num_layers):
            layer = self.layers[i]
            hidden_states = layer(
                hidden_states=hidden_states,
                position_ids=position_ids,
                kv_cache=kv_caches[i],
                attn_metadata=attn_metadata,
                layer_idx=i,
            )
        # Final layer norm.
        if self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states

    def llama_layer_forward_vllm(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        residual: Optional[torch.Tensor],
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            layer_idx=layer_idx,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def chatglm_layer_forward_vllm(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        layer_idx=0,
    ) -> torch.Tensor:
        # hidden_states: [num_tokens, h]
        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        attention_output = self.self_attention(
            hidden_states=layernorm_output,
            position_ids=position_ids,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            layer_idx=layer_idx,
        )

        # Residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = hidden_states
        layernorm_input = residual + attention_output
        # Layer norm post the self attention.
        layernorm_output = self.post_attention_layernorm(layernorm_input)
        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input
        output = self.mlp(layernorm_output) + residual
        return output

    def llama_attn_forward_vllm(
        vllm_version: str = "0.4.2",
    ):
        def llama_attn_forward_vllm(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata,
            layer_idx: int,
        ) -> torch.Tensor:
            qkv, _ = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            q, k = self.rotary_emb(positions, q, k)
            if "0.4.1" <= vllm_version <= "0.4.2":
                attn_output = self.attn(
                    q, k, v, kv_cache, attn_metadata, self.kv_scale, layer_idx
                )
            elif vllm_version >= "0.4.3":
                attn_output = self.attn(
                    q, k, v, kv_cache, attn_metadata, layer_idx=layer_idx
                )
            else:
                assert (
                    False
                ), "Only support 'vllm>=0.4.1'. Please update your vllm version."

            output, _ = self.o_proj(attn_output)
            return output

        return llama_attn_forward_vllm

    def chatglm_attn_forward_vllm(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        layer_idx: int = 0,
    ) -> torch.Tensor:
        qkv, _ = self.query_key_value(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(position_ids, q, k)
        context_layer = self.attn(
            q,
            k,
            v,
            kv_cache,
            attn_metadata,
            layer_idx=layer_idx,
        )
        attn_output, _ = self.dense(context_layer)
        return attn_output

    def update_module(m):
        if isinstance(m, Attention):
            m.forward = vllm_attn_forward.__get__(m, Attention)

            m = m.impl
            m_cls = m.__class__
            m.gather_last_q_vertical_slash_topk_vllm = (
                gather_last_q_vertical_slash_topk_vllm.__get__(m, m_cls)
            )
            m.forward = attn_forward.__get__(m, m_cls)
        if isinstance(m, LlamaDecoderLayer):
            m.forward = llama_layer_forward_vllm.__get__(m, LlamaDecoderLayer)
        if isinstance(m, LlamaModel):
            m.forward = llama_model_forward_vllm.__get__(m, LlamaModel)
        if isinstance(m, LlamaAttention):
            m.forward = llama_attn_forward_vllm(vllm_version).__get__(m, LlamaAttention)
        if isinstance(m, GLMBlock):
            m.forward = chatglm_layer_forward_vllm.__get__(m, GLMBlock)
        if isinstance(m, GLMTransformer):
            m.forward = chatglm_model_forward_vllm.__get__(m, GLMTransformer)
        if isinstance(m, GLMAttention):
            m.forward = chatglm_attn_forward_vllm.__get__(m, GLMAttention)

    return update_module


def minference_patch_vllm_tp(self, config_file, patch_config):    
    self.model_runner.model.apply(
        minference_patch_vllm_executor(config_file, patch_config)
    )


def patch_vllm_tp():
    from vllm.worker.worker import Worker
    Worker.minference_patch_vllm_tp = minference_patch_vllm_tp
