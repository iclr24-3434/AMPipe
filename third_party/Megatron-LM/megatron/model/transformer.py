# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Transformer."""
import math
from contextlib import nullcontext
import torch
import torch.nn.functional as F

from megatron import get_timers, get_args, core, get_num_microbatches
from .module import MegatronModule
from megatron.core import mpu, tensor_parallel, parallel_state
from megatron.model.enums import AttnMaskType, ModelType, LayerType, AttnType
from megatron.model import LayerNorm, megablocks_utils
from megatron.model.fused_softmax import FusedScaleMaskSoftmax
from megatron.model.fused_bias_gelu import bias_gelu_impl
from megatron.model.utils import attention_mask_func, openai_gelu, erf_gelu

try:
    from einops import rearrange
except ImportError:
    rearrange = None


from flash_attn.flash_helper import flash_attn_megablock_call
from megatron.model.ampipe import AttMoEPipe, XAttMoEPipe

import os
TIMERLEVEL=int(os.environ.get('TIMERLEVEL', 0))

def timer_wrapper_start(timers, name, level_on):
    global TIMERLEVEL
    if level_on >= TIMERLEVEL:
        timers(name, log_level=0).start(barrier=False)
def timer_wrapper_stop(timers, name, level_on):
    global TIMERLEVEL
    if level_on >= TIMERLEVEL:
        timers(name).stop()

class BlockEnter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, timers):
        timer_wrapper_start(timers, 'block', 1)
        ctx.timers = timers
        return hidden_states

    @staticmethod
    def backward(ctx, hidden_states_grad):
        timer_wrapper_stop(ctx.timers, 'block-bwd', 1)
        return hidden_states_grad, None

class BlockExit(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, timers):
        timer_wrapper_stop(timers, 'block', 1)
        ctx.timers = timers
        return hidden_states

    @staticmethod
    def backward(ctx, hidden_states_grad):
        timer_wrapper_start(ctx.timers, 'block-bwd', 1)
        return hidden_states_grad, None


""" We use the following notation throughout this file:
     h: hidden size
     n: number of attention heads
     p: number of model parallel partitions
     np: n/p
     hp: h/p
     hn: h/n
     b: batch size
     s: sequence length
     l: number of layers
    Transformer takes input of size [s, b, h] and returns a
    tensor of the same size. We use the following arguments:
        hyperparameters: transformer hyperparameters
"""

class DropPath(MegatronModule):
    """Drop paths (Stochastic Depth) per sample
    (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_state):
        if self.drop_prob == 0. or not self.training:
            return hidden_state
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        # hidden_state: [s, b, h]
        shape = (1,) + (hidden_state.shape[1],) + (1,) * (hidden_state.ndim - 2)
        random_tensor = keep_prob + \
            torch.rand(shape, dtype=hidden_state.dtype, device=hidden_state.device)
        random_tensor.floor_()  # binarize
        output = hidden_state.div(keep_prob) * random_tensor
        return output

def _args_to_kwargs():
    args = get_args()

    common_kwargs = {
        "params_dtype": args.params_dtype,
        "use_cpu_initialization": args.use_cpu_initialization,
        "perform_initialization": args.perform_initialization,
        "gradient_accumulation_fusion": args.gradient_accumulation_fusion,
        "sequence_parallel_enabled": args.sequence_parallel,
    }
    return common_kwargs

class ParallelMLP(MegatronModule):
    """MLP.

    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.
    """

    def __init__(self, init_method, output_layer_init_method):
        super(ParallelMLP, self).__init__()
        args = get_args()


        # Project to 4h.
        self.dense_h_to_4h = tensor_parallel.ColumnParallelLinear(
            args.hidden_size,
            args.ffn_hidden_size,
            gather_output=False,
            init_method=init_method,
            skip_bias_add=True,
            async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
            **_args_to_kwargs())

        self.bias_gelu_fusion = args.bias_gelu_fusion
        self.activation_func = F.gelu
        if args.openai_gelu:
            self.activation_func = openai_gelu
        elif args.onnx_safe:
            self.activation_func = erf_gelu

        # Project back to h.
        self.dense_4h_to_h = tensor_parallel.RowParallelLinear(
            args.ffn_hidden_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs())

    def forward(self, hidden_states):

        # [s, b, 4hp]
        intermediate_parallel, bias_parallel = self.dense_h_to_4h(hidden_states)

        if self.bias_gelu_fusion:
             intermediate_parallel = \
                     bias_gelu_impl(intermediate_parallel, bias_parallel)
        else:
            intermediate_parallel = \
                self.activation_func(intermediate_parallel + bias_parallel)

        # [s, b, h]
        output, output_bias = self.dense_4h_to_h(intermediate_parallel)
        return output, output_bias

class SwitchMLP(MegatronModule):
    """
    Routes input to one of N MLP "experts"
    """
    def __init__(self, init_method, output_layer_init_method):
        super(SwitchMLP, self).__init__()
        args = get_args()
        self.router = torch.nn.Linear(args.hidden_size, args.moe_num_experts)
        self.experts = torch.nn.ModuleList()
        for i in range(args.moe_num_experts):
            self.experts.append(ParallelMLP(init_method, output_layer_init_method))

    def forward(self, hidden_states):
        # hidden_states: [s, b, h]
        s = hidden_states.size(0)
        b = hidden_states.size(1)
        h = hidden_states.size(2)
        route = self.router(hidden_states)
        route = torch.nn.functional.softmax(route, dim=2)
        max_prob, max_ind = torch.max(route, dim=2)
        max_prob = torch.unsqueeze(max_prob, 2) # [s b 1]

        # TODO (rprenger) TODO this could be made easier to read
        # Converting [s, b, h] to [s*b, h].
        # Each vector could be routed differently
        hidden_states = hidden_states.view(-1, hidden_states.size(2)) # [s*b h]
        max_prob = max_prob.view(-1, max_prob.size(2)) # [s*b 1]
        max_ind = max_ind.view(-1) # [s*b]

        output_total = torch.empty_like(hidden_states)
        output_bias_total = torch.empty_like(hidden_states)
        #TODO (rprenger) This does each expert in serial, but it could be parallelized

        for expert_num, expert in enumerate(self.experts):
            local_indices = (max_ind == expert_num).nonzero()
            hidden = hidden_states[local_indices,:]
            output, output_bias = expert(hidden)
            output_bias = output_bias.expand_as(output)
            output_total[local_indices,:] = output
            output_bias_total[local_indices,:] = output_bias

        output_total = output_total*max_prob
        output_bias_total = output_bias_total*max_prob
        output_total = output_total.view(s, b, h)
        output_bias_total = output_bias_total.view(s, b, h)

        return output_total, output_bias_total

class _MegablocksAdapter(MegatronModule):

    def __init__(self, layer_cls, init_method, output_layer_init_method):
        super().__init__()
        megablocks_utils.assert_megablocks_is_available()
        args = megablocks_utils.arguments.from_megatron(get_args())
        args.device = torch.cuda.current_device()
        args.init_method = init_method
        args.output_layer_init_method = output_layer_init_method

        # NOTE: Shard the MoE layers over the data parallel group. Expert
        # parallel sharding and data parallel sharding could be decoupled
        # by extending the optimizer to handle data parallel reductions for
        # MoE and non-MoE parameters separately.
        if args.moe_expert_model_parallelism:
            args.expert_parallel_group = parallel_state.get_data_parallel_group()
        args.tensor_model_parallel_group = mpu.get_tensor_model_parallel_group()
        self.moe = layer_cls(args)

    def forward(self, x):
        return self.moe.tutel_forward(x, get_timers(), timer_wrapper_start, timer_wrapper_stop, [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])

class MoE(_MegablocksAdapter):

    def __init__(self, init_method, output_layer_init_method):
        megablocks_utils.assert_megablocks_is_available()
        super().__init__(
            megablocks_utils.moe.MoE, init_method, output_layer_init_method)

class dMoE(_MegablocksAdapter):

    def __init__(self, init_method, output_layer_init_method):
        megablocks_utils.assert_megablocks_is_available()
        super().__init__(
            megablocks_utils.dmoe.dMoE, init_method, output_layer_init_method)

class CoreAttention(MegatronModule):

    def __init__(self, layer_number,
                 attn_mask_type=AttnMaskType.padding):
        super(CoreAttention, self).__init__()
        args = get_args()
        self.fp16 = args.fp16
        self.bf16 = args.bf16

        self.apply_query_key_layer_scaling = args.apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = args.attention_softmax_in_fp32
        if self.apply_query_key_layer_scaling:
            self.attention_softmax_in_fp32 = True
        self.layer_number = max(1, layer_number)
        self.attn_mask_type = attn_mask_type
        self.sequence_parallel = args.sequence_parallel

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = core.utils.divide(projection_size,
                                                           world_size)
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        coeff = None
        self.norm_factor = math.sqrt(self.hidden_size_per_attention_head)
        if self.apply_query_key_layer_scaling:
            coeff = self.layer_number
            self.norm_factor *= coeff

        self.scale_mask_softmax = FusedScaleMaskSoftmax(
            self.fp16, self.bf16,
            self.attn_mask_type,
            args.masked_softmax_fusion,
            attention_mask_func,
            self.attention_softmax_in_fp32,
            coeff)

        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(args.attention_dropout)

    def forward(self, query_layer, key_layer,
                value_layer, attention_mask, offset=0):

        # ===================================
        # Raw attention scores. [b, np, s, s]
        # ===================================

        # [b, np, sq, sk]
        output_size = (query_layer.size(1),
                       query_layer.size(2),
                       query_layer.size(0),
                       key_layer.size(0))

        # [sq, b, np, hn] -> [sq, b * np, hn]
        query_layer = query_layer.view(output_size[2],
                                       output_size[0] * output_size[1], -1)
        # [sk, b, np, hn] -> [sk, b * np, hn]
        key_layer = key_layer.view(output_size[3],
                                   output_size[0] * output_size[1], -1)

        # preallocting input tensor: [b * np, sq, sk]
        matmul_input_buffer = mpu.get_global_memory_buffer().get_tensor(
            (output_size[0]*output_size[1], output_size[2], output_size[3]),
            query_layer.dtype, "mpu")

        # Raw attention scores. [b * np, sq, sk]
        matmul_result = torch.baddbmm(
            matmul_input_buffer,
            query_layer.transpose(0, 1),   # [b * np, sq, hn]
            key_layer.transpose(0, 1).transpose(1, 2),  # [b * np, hn, sk]
            beta=0.0, alpha=(1.0/self.norm_factor))

        # change view to [b, np, sq, sk]
        attention_scores = matmul_result.view(*output_size)

        # ===========================
        # Attention probs and dropout
        # ===========================

        # attention scores and attention mask [b, np, sq, sk]
        attention_probs = self.scale_mask_softmax(attention_scores,
                                                  attention_mask)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        if not self.sequence_parallel:
            with tensor_parallel.get_cuda_rng_tracker().fork():
                attention_probs = self.attention_dropout(attention_probs)
        else:
            attention_probs = self.attention_dropout(attention_probs)

        # =========================
        # Context layer. [sq, b, hp]
        # =========================

        # value_layer -> context layer.
        # [sk, b, np, hn] --> [b, np, sq, hn]

        # context layer shape: [b, np, sq, hn]
        output_size = (value_layer.size(1),
                       value_layer.size(2),
                       query_layer.size(0),
                       value_layer.size(3))

        # change view [sk, b * np, hn]
        value_layer = value_layer.view(value_layer.size(0),
                                       output_size[0] * output_size[1], -1)

        # change view [b * np, sq, sk]
        attention_probs = attention_probs.view(output_size[0] * output_size[1],
                                               output_size[2], -1)

        # matmul: [b * np, sq, hn]
        context_layer = torch.bmm(attention_probs, value_layer.transpose(0, 1))

        # change view [b, np, sq, hn]
        context_layer = context_layer.view(*output_size)

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class FlashSelfAttention(torch.nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """
    def __init__(self, causal=False, softmax_scale=None, attention_dropout=0.0,
                 device=None, dtype=None, fusion_split=-1, version=1):
        super().__init__()
        assert flash_attn_megablock_call is not None, ('Please install FlashAttention first, '
                                                      'e.g., with pip install flash-attn')
        assert rearrange is not None, 'Please install einops first, e.g., with pip install einops'
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout
        self.chunk_size = fusion_split
        self.version = 1

    def forward(self, q, k, v):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q, k, v: The tensor containing the query, key, and value. (B, S, H, D)
        """
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda
        batch_size, seqlen = q.shape[0], q.shape[1]

        if self.chunk_size == -1 or not self.training:
            q, k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [q, k, v]]
            max_s = seqlen
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                  device=q.device)
            output = flash_attn_megablock_call(
                q, k, v, cu_seqlens, cu_seqlens, max_s, max_s,
                self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=self.causal,
                version=self.version
            )
            output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            return output
        else:
            k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [k, v]]
            max_s = seqlen
            outs = []
            chunk_len = (seqlen + self.chunk_size - 1) // self.chunk_size
            base = 0
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                      device=q.device)
            
            for c in range(self.chunk_size):
                slen = min(seqlen - base, chunk_len)
                cu_seqlens_q = torch.arange(0, (batch_size + 1) * slen, step=slen, dtype=torch.int32,
                                      device=q.device)
                q_use = rearrange(q[:,base:base+slen], 'b s ... -> (b s) ...')

                assert self.causal 

                output_chunk = flash_attn_megablock_call(
                    q_use, k, v, cu_seqlens_q, cu_seqlens, slen, max_s,
                    self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=self.causal,
                    causal_q_offset=base if self.causal else 0, #fixed ,
                    version=self.version
                )
                outs.append(rearrange(output_chunk, '(b s) ... -> b s ...', b=batch_size))
                base += slen
            return outs
    
    

class ParallelAttention(MegatronModule):
    """Parallel self-attention layer abstract class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, init_method,
                 output_layer_init_method, layer_number,
                 attention_type=AttnType.self_attn,
                 attn_mask_type=AttnMaskType.padding):
        super(ParallelAttention, self).__init__()
        args = get_args()
        self.layer_number = max(1, layer_number)
        self.attention_type = attention_type
        self.attn_mask_type = attn_mask_type
        self.params_dtype = args.params_dtype
        self.sequence_parallel = args.sequence_parallel

        self.use_flash_attn = args.use_flash_attn
        if self.use_flash_attn:
            if flash_attn_megablock_call is None:
                raise ImportError('FlashAttention is not installed, please install with '
                                  'pip install flash-attn')
            assert attention_type == AttnType.self_attn, ('FlashAttention code path only supports '
                                                          'self-attention for now')
            assert self.attn_mask_type == AttnMaskType.causal, ('FlashAttention code path only '
                                                                'supports causal mask for now')
            if rearrange is None:
                raise ImportError('einops is not installed, please install with pip install einops')

        projection_size = args.kv_channels * args.num_attention_heads

        # Per attention head and per partition values.
        world_size = mpu.get_tensor_model_parallel_world_size()
        self.hidden_size_per_attention_head = core.utils.divide(
            projection_size, args.num_attention_heads)
        self.num_attention_heads_per_partition = core.utils.divide(
            args.num_attention_heads, world_size)

        # Strided linear layer.
        if attention_type == AttnType.self_attn:
            self.query_key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                3 * projection_size,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())
        else:
            assert attention_type == AttnType.cross_attn
            self.query = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                projection_size,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())


            self.key_value = tensor_parallel.ColumnParallelLinear(
                args.hidden_size,
                2 * projection_size,
                gather_output=False,
                init_method=init_method,
                async_tensor_model_parallel_allreduce=args.async_tensor_model_parallel_allreduce,
                **_args_to_kwargs())

        self.core_attention = CoreAttention(self.layer_number,
                                            self.attn_mask_type)
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'

        if self.use_flash_attn:
            self.core_attention_flash = FlashSelfAttention(
                causal=True, attention_dropout=args.attention_dropout, fusion_split=args.pipe_degree, version=1
            )
        else:
            self.chunk_size = args.fusion_split

        self.pipe_degree = args.pipe_degree
        # Output.
        self.dense = tensor_parallel.RowParallelLinear(
            projection_size,
            args.hidden_size,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            skip_bias_add=True,
            **_args_to_kwargs())

    def _checkpointed_attention_forward(self, query_layer, key_layer,
                                        value_layer, attention_mask):
        """Forward method with activation checkpointing."""
        def custom_forward(*inputs):
            query_layer = inputs[0]
            key_layer = inputs[1]
            value_layer = inputs[2]
            attention_mask = inputs[3]
            output_ = self.core_attention(query_layer, key_layer,
                                          value_layer, attention_mask)
            return output_

        hidden_states = tensor_parallel.checkpoint(
            custom_forward,
            False, query_layer, key_layer, value_layer, attention_mask)

        return hidden_states

    def _allocate_memory(self, inference_max_sequence_len, batch_size):
        return torch.empty(
            inference_max_sequence_len,
            batch_size,
            self.num_attention_heads_per_partition,
            self.hidden_size_per_attention_head,
            dtype=self.params_dtype,
            device=torch.cuda.current_device())

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, inference_params=None, skip_core=False):
        # hidden_states: [sq, b, h]

        # =================================================
        # Pre-allocate memory for key-values for inference.
        # =================================================
        if inference_params:
            if self.layer_number not in inference_params.key_value_memory_dict:
                inf_max_seq_len = inference_params.max_sequence_len
                inf_max_batch_size = inference_params.max_batch_size
                inference_key_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_value_memory = self._allocate_memory(
                    inf_max_seq_len, inf_max_batch_size)
                inference_params.key_value_memory_dict[self.layer_number] = (
                    inference_key_memory, inference_value_memory)
            else:
                inference_key_memory, inference_value_memory = \
                    inference_params.key_value_memory_dict[self.layer_number]

        # =====================
        # Query, Key, and Value
        # =====================

        if self.attention_type == AttnType.self_attn:
            # Attention heads [sq, b, h] --> [sq, b, (np * 3 * hn)]
            mixed_x_layer, _ = self.query_key_value(hidden_states)

            # [sq, b, (np * 3 * hn)] --> [sq, b, np, 3 * hn]
            new_tensor_shape = mixed_x_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 3 * self.hidden_size_per_attention_head)
            mixed_x_layer = mixed_x_layer.view(*new_tensor_shape)

            # [sq, b, np, 3 * hn] --> 3 [sq, b, np, hn]
            (query_layer,
             key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_x_layer, 3)
        else:
            # Attention heads [sk, b, h] --> [sk, b, (np * 2 * hn)]
            mixed_kv_layer, _ = self.key_value(encoder_output)

            # [sk, b, (np * 2 * hn)] --> [sk, b, np, 2 * hn]
            new_tensor_shape = mixed_kv_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 2 * self.hidden_size_per_attention_head)
            mixed_kv_layer = mixed_kv_layer.view(*new_tensor_shape)

            # [sk, b, np, 2 * hn] --> 2 [sk, b, np, hn]
            (key_layer,
             value_layer) = tensor_parallel.split_tensor_along_last_dim(mixed_kv_layer, 2)

            # Attention head [sq, b, h] --> [sq, b, hp]
            query_layer, _ = self.query(hidden_states)
            # [sq, b, hp] --> [sq, b, np, hn]
            new_tensor_shape = query_layer.size()[:-1] + \
                (self.num_attention_heads_per_partition,
                 self.hidden_size_per_attention_head)
            query_layer = query_layer.view(*new_tensor_shape)

        # ==================================
        # Adjust key and value for inference
        # ==================================

        if inference_params:
            batch_start = inference_params.batch_size_offset
            batch_end = batch_start + key_layer.size(1)
            assert batch_end <= inference_key_memory.size(1)
            sequence_start = inference_params.sequence_len_offset
            sequence_end = sequence_start + key_layer.size(0)
            assert sequence_end <= inference_key_memory.size(0)
            # Copy key and values.
            inference_key_memory[sequence_start:sequence_end,
                                 batch_start:batch_end, ...] = key_layer
            inference_value_memory[sequence_start:sequence_end,
                                   batch_start:batch_end, ...] = value_layer
            key_layer = inference_key_memory[
                :sequence_end, batch_start:batch_end, ...]
            value_layer = inference_value_memory[
                :sequence_end, batch_start:batch_end, ...]

        # ==================================
        # core attention computation
        # ==================================

        if skip_core:
            assert not self.sequence_parallel
            assert not self.checkpoint_core_attention

        if not self.use_flash_attn:
            if self.checkpoint_core_attention:
                context_layer = self._checkpointed_attention_forward(
                    query_layer, key_layer, value_layer, attention_mask)
            else:
                if skip_core:
                    return query_layer, key_layer, value_layer
                context_layer = self.core_attention(
                    query_layer, key_layer, value_layer, attention_mask)
        elif self.pipe_degree == -1 or not self.training:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
            if not self.sequence_parallel:
                if skip_core:
                    return q, k, v
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    timers = get_timers()
                    timer_wrapper_start(timers, 'attn', 0)
                    #timers('attn', log_level=0).start(barrier=True)
                    context_layer = self.core_attention_flash(q, k, v)
                    timer_wrapper_stop(timers, 'attn', 0)
            else:
                context_layer = self.core_attention_flash(q, k, v)
            context_layer = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
        else:
            q, k, v = [rearrange(x, 's b ... -> b s ...').contiguous()
                       for x in (query_layer, key_layer, value_layer)]
            assert not self.sequence_parallel
            
            if skip_core:
                return q, k, v
            timers = get_timers()
            timer_wrapper_start(timers, 'attn', 0)
            with tensor_parallel.get_cuda_rng_tracker().fork():
                context_layers = self.core_attention_flash(q, k, v) 
            timer_wrapper_stop(timers, 'attn', 0)
            context_layers = [rearrange(context_layer, 'b s h d -> s b (h d)').contiguous() for context_layer in context_layers]

            outs = []
            for context_layer in context_layers:
                output, bias = self.dense(context_layer)
                outs.append(output)
            #print("outs :" , len(outs))
            return outs, bias

        # =================
        # Output. [sq, b, h]
        # =================

        output, bias = self.dense(context_layer)

        return output, bias


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out


def get_bias_dropout_add(training):
    def _bias_dropout_add(x, bias, residual, prob):
        return bias_dropout_add(x, bias, residual, prob, training)
    return _bias_dropout_add


@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: torch.Tensor,
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)


@torch.jit.script
def bias_dropout_add_fused_inference(x: torch.Tensor,
                                     bias: torch.Tensor,
                                     residual: torch.Tensor,
                                     prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, False)


def attdense(q_use, k, v, base, batch_size, cu_seqlens, max_s, slen, chunk_len, attn, self_attention):
          
    cu_seqlens_q = torch.arange(0, (batch_size + 1) * slen, step=slen, dtype=torch.int32,
                                      device=q_use.device)
    output_chunk = flash_attn_megablock_call(
                    q_use, k, v, cu_seqlens_q, cu_seqlens, slen, max_s,
                    attn.dropout_p if attn.training else 0.0,
                    softmax_scale=attn.softmax_scale, causal=attn.causal,
                    causal_q_offset=base if attn.causal else 0, #fixed 
                    version=1
                )


    context_layer = rearrange(output_chunk, '(b s) ... -> b s ...', b=batch_size)
    
    context_layer_tmp = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()

            
    attention_output, attention_bias = self_attention.dense(context_layer_tmp)

    return attention_output, attention_bias

class BWDDEBUG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, info):
        ctx.info = info 
        return inp

    @staticmethod
    def backward(ctx, grad_inp):
        #if torch.distributed.get_rank() == 0:
        #    print("CALLING BWD: ", ctx.info)
        return grad_inp, None 
    

class FusionAttMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, hidden_states, others):
        
        attn, self_attention, hidden_dropout, post_attention_layernorm, moe = others

        ctx.attn, ctx.self_attention = attn, self_attention
        attention_outputs = []
        batch_size, seqlen = q.shape[0], q.shape[1]
        
        k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [k, v]]
        k.requires_grad = True 
        v.requires_grad = True
      
        ctx.k, ctx.v = k, v

        max_s = seqlen
            #context_layers = []
        chunk_len = (seqlen + attn.chunk_size - 1) // attn.chunk_size
        
        ctx.batch_size, ctx.seqlen, ctx.max_s, ctx.chunk_len = batch_size, seqlen, max_s, chunk_len
        base = 0

        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                      device=q.device)

        hidden_states = hidden_states.chunk(attn.chunk_size)
        ctx.hidden_states = hidden_states
        ctx.q_uses = []
        layernorm_inputs = []
        layernorm_input_xs = []

        mlp_outputs = []

        moe_inps = [[],[],[]]
        moe_scores = [[],[],[]]
        first_a2a = [[], []]
        second_a2a = [[], []]
        a2a_comp = [[], []]
        enter_a2a = [[], []]
        bias_dropout_add_func = bias_dropout_add_fused_train

        torch.cuda.nvtx.range_push("FWD Fusion")

        for c in range(attn.chunk_size):
            slen = min(seqlen - base, chunk_len) 
            #q_use.requires_grad = True
            
            q_use = rearrange(q[:,base:base+slen], 'b s ... -> (b s) ...')
            q_use.requires_grad = True

            hidden_states[c].requires_grad = True

            with torch.enable_grad():                

                attention_output, attention_bias = attdense(q_use, k, v, base, batch_size, cu_seqlens, max_s, slen, chunk_len, attn, self_attention)
                layernorm_input = bias_dropout_add_func(
                        attention_output,
                        attention_bias.expand_as(hidden_states[c]),
                        hidden_states[c],
                        hidden_dropout)
            
            layernorm_input_x = layernorm_input.detach()

            layernorm_input_xs.append(layernorm_input_x)
            layernorm_input_x.requires_grad = True
            with torch.enable_grad():   
                layernorm_input_x = BWDDEBUG.apply(layernorm_input_x, "layernorm_input_x")
                layernorm_output = post_attention_layernorm(layernorm_input_x)
                #moe_inp = BWDDEBUG.apply(layernorm_output, "layernorm_output")
                moe_inp = layernorm_output.detach().requires_grad_()
                #mlp_output, _ = mlp(lay0)
                router_inp = layernorm_output.detach().requires_grad_()

                scores, expert_weights, top_experts, scoresx, scores_origin = moe.router.forwardx(router_inp) #bwd: moe_inp get
                moe_inps[0].append(layernorm_output)
                moe_inps[1].append(moe_inp)
                moe_inps[2].append(router_inp)

                moe_scores[0].append(scores)
                moe_scores[1].append(scoresx)
                moe_scores[2].append(scores_origin)
            
            torch.cuda.nvtx.range_push(f"FWDMoE-{c}")
            with torch.enable_grad(): 
                sl, bs, hs = moe_inp.size()

                moe_inp = BWDDEBUG.apply(moe_inp, "moe_inp")
                #x, tokens_per_expert = moe.forward_fn(moe_inp, top_experts)

                x, recv_counts, send_counts, parallel_tokens_per_expert, \
            parallel_indices, parallel_bin_ids, parallel_bins, expert_capacity, \
            indices, bin_ids, bins, tokens_per_expert = moe.parallel_forward_prepare(moe_inp, top_experts)

                enter_a2a[0].append(x)
                x = x.detach().requires_grad_()
                enter_a2a[1].append(x)

                parallel_x = moe.parallel_forward_a2a1( x, recv_counts, send_counts)

                first_a2a[0].append(parallel_x)
                parallel_x = parallel_x.detach().requires_grad_()
                first_a2a[1].append(parallel_x)

                parallel_x = moe.parallel_forward_compute(parallel_x, parallel_tokens_per_expert,
            parallel_indices, parallel_bin_ids, parallel_bins, expert_capacity)

                a2a_comp[0].append(parallel_x)
                parallel_x = parallel_x.detach().requires_grad_()
                a2a_comp[1].append(parallel_x)

                x = moe.parallel_forward_a2a2(parallel_x, send_counts, recv_counts)

                second_a2a[0].append(x)
                x = x.detach().requires_grad_()
                second_a2a[1].append(x)

                x, tokens_per_expert = moe.parallel_forward_post(x, indices, bin_ids, bins, tokens_per_expert)

                x = BWDDEBUG.apply(x, "moe fwd")
                x = x * expert_weights.view(-1, 1)

                megablocks_utils.save_load_balancing_loss((tokens_per_expert, scores), moe.moe_id)

                mlp_output = x.view(sl, bs, hs)

                mlp_output = BWDDEBUG.apply(mlp_output, "mlp_output")

                mlp_outputs.append(mlp_output)  

            torch.cuda.nvtx.range_pop()
                        
            ctx.q_uses.append(q_use)
            base += slen
        
            layernorm_inputs.append(layernorm_input)
        #if torch.distributed.get_rank() == 0:
        #    print(ctx.q_uses[0].requires_grad, ctx.k.requires_grad, ctx.v.requires_grad)
        ctx.layernorm_inputs = layernorm_inputs
        ctx.layernorm_input_xs = layernorm_input_xs
        ctx.mlp_outputs = mlp_outputs
        ctx.moe_inps = moe_inps
        ctx.moe_scores = moe_scores
        ctx.first_a2a = first_a2a
        ctx.second_a2a = second_a2a
        ctx.enter_a2a = enter_a2a
        ctx.a2a_comp = a2a_comp


        mlp_ret = torch.cat(mlp_outputs)
        lay_ret = torch.cat(layernorm_inputs)

        torch.cuda.nvtx.range_pop()

        #ctx.attention_output = attention_output
        return lay_ret, mlp_ret
    
    @staticmethod
    def backward(ctx, grad_layernorm_input, grad_mlp_output):
        #if torch.distributed.get_rank() == 0:
        #    print("bwd0")
        attn, self_attention = ctx.attn, ctx.self_attention
        grad_layernorm_inputs = grad_layernorm_input.chunk(attn.chunk_size)
        grad_mlp_outputs = list(grad_mlp_output.chunk(attn.chunk_size))

        batch_size, seqlen, max_s, chunk_len = ctx.batch_size, ctx.seqlen, ctx.max_s, ctx.chunk_len 

        base = 0

        torch.cuda.nvtx.range_push("BWD Fusion")

        for c in range(attn.chunk_size):
            with torch.enable_grad():
                ctx.mlp_outputs[c].backward(grad_mlp_outputs[c])
                grad_mlp_outputs[c] = None

            torch.cuda.nvtx.range_push(f"MoE-{c}")
            with torch.enable_grad():
                ctx.second_a2a[0][c].backward(ctx.second_a2a[1][c].grad)
                del ctx.second_a2a[1][c].grad
                ctx.a2a_comp[0][c].backward(ctx.a2a_comp[1][c].grad)
                del ctx.a2a_comp[1][c].grad
                ctx.first_a2a[0][c].backward(ctx.first_a2a[1][c].grad)
                del ctx.first_a2a[1][c].grad
                ctx.enter_a2a[0][c].backward(ctx.enter_a2a[1][c].grad)
                del ctx.enter_a2a[1][c].grad
            torch.cuda.nvtx.range_pop()
            #if torch.distributed.get_rank() == 0:
            #    print(ctx.scores_x_list[c].grad.norm())
            #if torch.distributed.get_rank() == 0:
            #    print("exec ", c)
            with torch.enable_grad():
                ctx.moe_scores[2][c].backward(ctx.moe_scores[0][c].grad + ctx.moe_scores[1][c].grad)
                del ctx.moe_scores[0][c].grad, ctx.moe_scores[1][c].grad 
                ctx.moe_inps[0][c].backward(ctx.moe_inps[1][c].grad + ctx.moe_inps[2][c].grad)
                del ctx.moe_inps[1][c].grad, ctx.moe_inps[2][c].grad
                
            with torch.enable_grad():
                ctx.layernorm_inputs[c].backward(grad_layernorm_inputs[c] + ctx.layernorm_input_xs[c].grad)

        torch.cuda.nvtx.range_pop()

        #if torch.distributed.get_rank() == 0:
        #    print("fetch: ", id(ctx.q), id(ctx.k), id(ctx.v))
        #    print("grads: ", ctx.q.grad, ctx.k.grad.norm(), ctx.v.grad.norm(), ctx.q_uses[0].grad.norm())
        q_grads = [rearrange(q.grad, '(b s) ... -> b s ...', b=ctx.batch_size) for q in ctx.q_uses]

        q_grad = torch.cat(q_grads, dim=1).contiguous()
        k_grad = rearrange(ctx.k.grad, '(b s) ... -> b s ...', b=ctx.batch_size).contiguous()
        v_grad = rearrange(ctx.v.grad, '(b s) ... -> b s ...', b=ctx.batch_size).contiguous()
        h_grad = torch.cat([h.grad for h in ctx.hidden_states])

        return q_grad, k_grad, v_grad, h_grad, None

streams = {}
def get_comp(dev=None):
    return torch.cuda.current_stream()


def get_comp0(dev=None):
    #return torch.cuda.current_stream()
    if 'h' in streams:
        return streams['h']
    streams['h'] = torch.cuda.Stream()
    return streams['h']

def get_comm(dev=None):
    #return torch.cuda.current_stream()
    if 'm' in streams:
        return streams['m']
    streams['m'] = torch.cuda.Stream()
    return streams['m']
import numba


class TutelFusionAttMoE(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v, hidden_states, others):
        attn, self_attention, hidden_dropout, post_attention_layernorm, moe = others

        ctx.attn, ctx.self_attention = attn, self_attention
        attention_outputs = []
        batch_size, seqlen = q.shape[0], q.shape[1]
        
        k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [k, v]]
        k.requires_grad = True 
        v.requires_grad = True
      
        ctx.k, ctx.v = k, v

        max_s = seqlen
            #context_layers = []
        chunk_len = (seqlen + attn.chunk_size - 1) // attn.chunk_size
        
        ctx.batch_size, ctx.seqlen, ctx.max_s, ctx.chunk_len = batch_size, seqlen, max_s, chunk_len
        base = 0

        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                      device=q.device)

        hidden_states = hidden_states.chunk(attn.chunk_size)
        ctx.hidden_states = hidden_states
        ctx.q_uses = []
        layernorm_inputs = []
        layernorm_input_xs = []

        mlp_outputs = []

        moe_inps = [[],[],[]]
        moe_scores = [[],[]]
        first_a2a = [[], []]
        second_a2a = [[], []]
        a2a_comp = [[], []]
        enter_a2a = [[], []]

        tokens_per_experts = []
        crits = []
        bias_dropout_add_func = bias_dropout_add_fused_train
        #print("device: ", torch.cuda.current_device())
        get_comp0().wait_stream(torch.cuda.current_stream())
        #torch.cuda.current_stream().wait_stream(get_comp())
        attn_events = []

        a2a1_events = []
        comp_events = []
        a2a2_events = []

        #torch.cuda.synchronize()
        for c in range(attn.chunk_size):
            with torch.cuda.stream(get_comp0()):

                slen = min(seqlen - base, chunk_len) 

                q_use = rearrange(q[:,base:base+slen], 'b s ... -> (b s) ...')
                q_use.requires_grad = True

                hidden_states[c].requires_grad = True

                with torch.enable_grad():                
                    attention_output, attention_bias = attdense(q_use, k, v, base, batch_size, cu_seqlens, max_s, slen, chunk_len, attn, self_attention)
                    layernorm_input = bias_dropout_add_func(
                        attention_output,
                        attention_bias.expand_as(hidden_states[c]),
                        hidden_states[c],
                        hidden_dropout)
            
                layernorm_input_x = layernorm_input.detach()

                layernorm_input_xs.append(layernorm_input_x)
                layernorm_input_x.requires_grad = True

                ctx.q_uses.append(q_use)
                base += slen
        
                layernorm_inputs.append(layernorm_input)
                with torch.enable_grad():   
                    layernorm_output = post_attention_layernorm(layernorm_input_x)
                #moe_inp = BWDDEBUG.apply(layernorm_output, "layernorm_output")
                    moe_inp = layernorm_output.detach().requires_grad_()
                #mlp_output, _ = mlp(lay0)
                    router_inp = layernorm_output.detach().requires_grad_()

                    scores, scores_origin = moe.router.tutel_forwardx(router_inp) #bwd: moe_inp get
                    moe_inps[0].append(layernorm_output)
                    moe_inps[1].append(moe_inp)
                    moe_inps[2].append(router_inp)

                    moe_scores[0].append(scores)
                    moe_scores[1].append(scores_origin)

                    sl, bs, hs = moe_inps[1][c].size()

                    x, tokens_per_expert, crit = moe.tutel_prepare(moe_inps[1][c], moe_scores[0][c])
                    tokens_per_experts.append(tokens_per_expert)
                    crits.append(crit)
                    enter_a2a[0].append(x)
                    x = x.detach().requires_grad_()
                    enter_a2a[1].append(x)

                    attn_events.append(torch.cuda.current_stream().record_event())
                    #attn_events[-1].record()

        #torch.cuda.synchronize()
        #get_comm().wait_stream(get_comp())
        
        #for c in range(attn.chunk_size):
            with torch.enable_grad(): 
                with torch.cuda.stream(get_comm()):

                    torch.cuda.current_stream().wait_event(attn_events[c])
                    
                    #attn_events[c].wait()
                    parallel_x = moe.tutel_a2a1(enter_a2a[1][c])

                    first_a2a[0].append(parallel_x)
                    parallel_x = parallel_x.detach().requires_grad_()
                    first_a2a[1].append(parallel_x)
                    
                    a2a1_events.append(torch.cuda.current_stream().record_event())
                    #a2a1_events[-1].record()

        
        
        for c in range(attn.chunk_size):
            with torch.enable_grad(): 
                with torch.cuda.stream(get_comp0()):
                    torch.cuda.current_stream().wait_event(a2a1_events[c])
                    parallel_x = moe.mlp(first_a2a[1][c])

                    a2a_comp[0].append(parallel_x)
                    parallel_x = parallel_x.detach().requires_grad_()
                    a2a_comp[1].append(parallel_x)

                    comp_events.append(torch.cuda.current_stream().record_event())
                    #comp_events[-1].record()
        #get_comm().wait_stream(get_comp())

        
        #torch.cuda.synchronize()
        for c in range(attn.chunk_size):
            with torch.enable_grad(): 
                with torch.cuda.stream(get_comm()):

                    torch.cuda.current_stream().wait_event(comp_events[c])
                    #comp_events[c].wait()
                    x = moe.tutel_a2a2(a2a_comp[1][c])

                    second_a2a[0].append(x)
                    x = x.detach().requires_grad_()
                    second_a2a[1].append(x)

                    a2a2_events.append(torch.cuda.current_stream().record_event())
                    #a2a2_events[-1].record()

        #torch.cuda.synchronize()
        #for c in range(attn.chunk_size):
            with torch.enable_grad(): 
                with torch.cuda.stream(get_comp0()):
                    #a2a2_events[c].wait()
                    torch.cuda.current_stream().wait_event(a2a2_events[c])
                    
                    x = moe.tutel_post(second_a2a[1][c], crits[c], moe_scores[0][c].dtype)

                    megablocks_utils.save_load_balancing_loss((tokens_per_experts[c], moe_scores[0][c]), moe.moe_id)

                    mlp_output = x.view(sl, bs, hs)

                    mlp_outputs.append(mlp_output)  
                        
        #torch.cuda.synchronize()
        #get_comp0().wait_stream(torch.cuda.current_stream())
        torch.cuda.current_stream().wait_stream(get_comp0())
        #if torch.distributed.get_rank() == 0:
        #    print(ctx.q_uses[0].requires_grad, ctx.k.requires_grad, ctx.v.requires_grad)
        ctx.layernorm_inputs = layernorm_inputs
        ctx.layernorm_input_xs = layernorm_input_xs
        ctx.mlp_outputs = mlp_outputs
        ctx.moe_inps = moe_inps
        ctx.moe_scores = moe_scores
        ctx.first_a2a = first_a2a
        ctx.second_a2a = second_a2a
        ctx.enter_a2a = enter_a2a
        ctx.a2a_comp = a2a_comp


        mlp_ret = torch.cat(mlp_outputs)
        lay_ret = torch.cat(layernorm_inputs)

        #torch.cuda.nvtx.range_pop()

        #ctx.attention_output = attention_output
        return lay_ret, mlp_ret

    @staticmethod
    def backward(ctx, grad_layernorm_input, grad_mlp_output):
        #if torch.distributed.get_rank() == 0:
        #    print("bwd0")
        attn, self_attention = ctx.attn, ctx.self_attention
        grad_layernorm_inputs = grad_layernorm_input.chunk(attn.chunk_size)
        grad_mlp_outputs = list(grad_mlp_output.chunk(attn.chunk_size))

        batch_size, seqlen, max_s, chunk_len = ctx.batch_size, ctx.seqlen, ctx.max_s, ctx.chunk_len 

        base = 0

        #torch.cuda.nvtx.range_push("BWD Fusion")
        #torch.cuda.synchronize()
        post_events = []
        #torch.cuda.current_stream().wait_stream(get_comp())
        get_comp0().wait_stream(torch.cuda.current_stream())

        for c in range(attn.chunk_size):
            with torch.enable_grad():
                with torch.cuda.stream(get_comp0()):
                    ctx.mlp_outputs[c].backward(grad_mlp_outputs[c])
                    grad_mlp_outputs[c] = None

                    #attn_events.append()

                    post_events.append(torch.cuda.current_stream().record_event())
                    #post_events[-1].record()
        #torch.cuda.synchronize()

        a2a2_events = []
        for c in range(attn.chunk_size):
            with torch.enable_grad():
                with torch.cuda.stream(get_comm()):

                    torch.cuda.current_stream().wait_event(post_events[c])
                    #post_events[c].wait()
                    ctx.second_a2a[0][c].backward(ctx.second_a2a[1][c].grad)
                    del ctx.second_a2a[1][c].grad

                    a2a2_events.append(torch.cuda.current_stream().record_event())
                    #a2a2_events[-1].record()

        comp_events = []
        for c in range(attn.chunk_size):
            with torch.enable_grad():
                with torch.cuda.stream(get_comp0()):
                    #a2a2_events[c].wait()
                    torch.cuda.current_stream().wait_event(a2a2_events[c])
                    
                    ctx.a2a_comp[0][c].backward(ctx.a2a_comp[1][c].grad)
                    del ctx.a2a_comp[1][c].grad

                    comp_events.append(torch.cuda.current_stream().record_event())
                    #comp_events[-1].record()

        #torch.cuda.synchronize()
        a2a1_events = []
        for c in range(attn.chunk_size):
            with torch.enable_grad():
                with torch.cuda.stream(get_comm()):
                    #comp_events[c].wait()

                    torch.cuda.current_stream().wait_event(comp_events[c])

                    ctx.first_a2a[0][c].backward(ctx.first_a2a[1][c].grad)
                    del ctx.first_a2a[1][c].grad

                    a2a1_events.append(torch.cuda.current_stream().record_event())
                    #a2a1_events[-1].record()
        #torch.cuda.synchronize()

        for c in range(attn.chunk_size):
            with torch.enable_grad():
                with torch.cuda.stream(get_comp0()):

                    torch.cuda.current_stream().wait_event(a2a1_events[c])
                    #a2a1_events[c].wait()
                    ctx.enter_a2a[0][c].backward(ctx.enter_a2a[1][c].grad)
                    del ctx.enter_a2a[1][c].grad

                    ctx.moe_scores[1][c].backward(ctx.moe_scores[0][c].grad)
                    del ctx.moe_scores[0][c].grad 
                    ctx.moe_inps[0][c].backward(ctx.moe_inps[1][c].grad + ctx.moe_inps[2][c].grad)
                    del ctx.moe_inps[1][c].grad, ctx.moe_inps[2][c].grad
                
            #with torch.enable_grad():
                    ctx.layernorm_inputs[c].backward(grad_layernorm_inputs[c] + ctx.layernorm_input_xs[c].grad)

        #get_comp().wait_stream(torch.cuda.current_stream())
        #torch.cuda.nvtx.range_pop()
        torch.cuda.current_stream().wait_stream(get_comp0())
        #if torch.distributed.get_rank() == 0:
        #    print("fetch: ", id(ctx.q), id(ctx.k), id(ctx.v))
        #    print("grads: ", ctx.q.grad, ctx.k.grad.norm(), ctx.v.grad.norm(), ctx.q_uses[0].grad.norm())
        q_grads = [rearrange(q.grad, '(b s) ... -> b s ...', b=ctx.batch_size) for q in ctx.q_uses]

        q_grad = torch.cat(q_grads, dim=1).contiguous()
        k_grad = rearrange(ctx.k.grad, '(b s) ... -> b s ...', b=ctx.batch_size).contiguous()
        v_grad = rearrange(ctx.v.grad, '(b s) ... -> b s ...', b=ctx.batch_size).contiguous()
        h_grad = torch.cat([h.grad for h in ctx.hidden_states])

        return q_grad, k_grad, v_grad, h_grad, None

fusion_g = []
class TutelFusionAttMoENormal(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, q, k, v, hidden_states, attention_mask, others):
        attn, self_attention, hidden_dropout, post_attention_layernorm, moe = others

        ctx.attn, ctx.self_attention = attn, self_attention
        attention_outputs = []
        batch_size, seqlen = q.shape[1], q.shape[0]
        
        #k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [k, v]]
        #k.requires_grad = True 
        #v.requires_grad = True
      
        ctx.k, ctx.v = k, v

        max_s = seqlen
            #context_layers = []
        chunk_len = (seqlen + attn.chunk_size - 1) // attn.chunk_size
        
        ctx.batch_size, ctx.seqlen, ctx.max_s, ctx.chunk_len = batch_size, seqlen, max_s, chunk_len
        base = 0

        cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                      device=q.device)

        hidden_states = hidden_states.chunk(attn.chunk_size)
        ctx.hidden_states = hidden_states
        ctx.q_uses = []
        layernorm_inputs = []
        layernorm_input_xs = []

        mlp_outputs = []

        moe_inps = [[],[],[]]
        moe_scores = [[],[]]
        first_a2a = [[], []]
        second_a2a = [[], []]
        a2a_comp = [[], []]
        enter_a2a = [[], []]

        tokens_per_experts = []
        crits = []
        bias_dropout_add_func = bias_dropout_add_fused_train
        #print("device: ", torch.cuda.current_device())
        get_comp0().wait_stream(torch.cuda.current_stream())
        #torch.cuda.current_stream().wait_stream(get_comp())
        attn_events = []

        a2a1_events = []
        comp_events = []
        a2a2_events = []

        #torch.cuda.synchronize()
        for c in range(attn.chunk_size):
            with torch.cuda.stream(get_comp0()):

                slen = min(seqlen - base, chunk_len) 

                #q_use = rearrange(q[:,base:base+slen], 'b s ... -> (b s) ...')
                #q_use.requires_grad = True

                hidden_states[c].requires_grad = True

                print("Q: ", q.size(), k.size(), v.size(), attention_mask.size())
                with torch.enable_grad():                
                    context_layer_tmp = self_attention.core_attention(
                        q[base:base+slen], k, v, attention_mask, offset=base)

                    attention_output, attention_bias = self_attention.dense(context_layer_tmp)
                    '''
                    attention_output, attention_bias = attdense(q_use, k, v, base, batch_size, cu_seqlens, max_s, slen, chunk_len, attn, self_attention)
                    '''
                    layernorm_input = bias_dropout_add_func(
                        attention_output,
                        attention_bias.expand_as(hidden_states[c]),
                        hidden_states[c],
                        hidden_dropout)
            
                layernorm_input_x = layernorm_input.detach()

                layernorm_input_xs.append(layernorm_input_x)
                layernorm_input_x.requires_grad = True

                ctx.q_uses.append(q_use)
                base += slen
        
                layernorm_inputs.append(layernorm_input)
                with torch.enable_grad():   
                    layernorm_output = post_attention_layernorm(layernorm_input_x)
                #moe_inp = BWDDEBUG.apply(layernorm_output, "layernorm_output")
                    moe_inp = layernorm_output.detach().requires_grad_()
                #mlp_output, _ = mlp(lay0)
                    router_inp = layernorm_output.detach().requires_grad_()

                    scores, scores_origin = moe.router.tutel_forwardx(router_inp) #bwd: moe_inp get
                    moe_inps[0].append(layernorm_output)
                    moe_inps[1].append(moe_inp)
                    moe_inps[2].append(router_inp)

                    moe_scores[0].append(scores)
                    moe_scores[1].append(scores_origin)

                    sl, bs, hs = moe_inps[1][c].size()

                    x, tokens_per_expert, crit = moe.tutel_prepare(moe_inps[1][c], moe_scores[0][c])
                    tokens_per_experts.append(tokens_per_expert)
                    crits.append(crit)
                    enter_a2a[0].append(x)
                    x = x.detach().requires_grad_()
                    enter_a2a[1].append(x)

                    attn_events.append(torch.cuda.current_stream().record_event())
                    #attn_events[-1].record()

        #torch.cuda.synchronize()
        #get_comm().wait_stream(get_comp())
        
        #for c in range(attn.chunk_size):
            with torch.enable_grad(): 
                with torch.cuda.stream(get_comm()):

                    torch.cuda.current_stream().wait_event(attn_events[c])
                    
                    #attn_events[c].wait()
                    parallel_x = moe.tutel_a2a1(enter_a2a[1][c])

                    first_a2a[0].append(parallel_x)
                    parallel_x = parallel_x.detach().requires_grad_()
                    first_a2a[1].append(parallel_x)
                    
                    a2a1_events.append(torch.cuda.current_stream().record_event())
                    #a2a1_events[-1].record()

        
        
        for c in range(attn.chunk_size):
            with torch.enable_grad(): 
                with torch.cuda.stream(get_comp0()):
                    torch.cuda.current_stream().wait_event(a2a1_events[c])
                    parallel_x = moe.mlp(first_a2a[1][c])

                    a2a_comp[0].append(parallel_x)
                    parallel_x = parallel_x.detach().requires_grad_()
                    a2a_comp[1].append(parallel_x)

                    comp_events.append(torch.cuda.current_stream().record_event())
                    #comp_events[-1].record()
        #get_comm().wait_stream(get_comp())

        
        #torch.cuda.synchronize()
        for c in range(attn.chunk_size):
            with torch.enable_grad(): 
                with torch.cuda.stream(get_comm()):

                    torch.cuda.current_stream().wait_event(comp_events[c])
                    #comp_events[c].wait()
                    x = moe.tutel_a2a2(a2a_comp[1][c])

                    second_a2a[0].append(x)
                    x = x.detach().requires_grad_()
                    second_a2a[1].append(x)

                    a2a2_events.append(torch.cuda.current_stream().record_event())
                    #a2a2_events[-1].record()

        #torch.cuda.synchronize()
        #for c in range(attn.chunk_size):
            with torch.enable_grad(): 
                with torch.cuda.stream(get_comp0()):
                    #a2a2_events[c].wait()
                    torch.cuda.current_stream().wait_event(a2a2_events[c])
                    
                    x = moe.tutel_post(second_a2a[1][c], crits[c], moe_scores[0][c].dtype)

                    megablocks_utils.save_load_balancing_loss((tokens_per_experts[c], moe_scores[0][c]), moe.moe_id)

                    mlp_output = x.view(sl, bs, hs)

                    mlp_outputs.append(mlp_output)  
                        
        #torch.cuda.synchronize()
        #get_comp0().wait_stream(torch.cuda.current_stream())
        torch.cuda.current_stream().wait_stream(get_comp0())
        #if torch.distributed.get_rank() == 0:
        #    print(ctx.q_uses[0].requires_grad, ctx.k.requires_grad, ctx.v.requires_grad)
        ctx.layernorm_inputs = layernorm_inputs
        ctx.layernorm_input_xs = layernorm_input_xs
        ctx.mlp_outputs = mlp_outputs
        ctx.moe_inps = moe_inps
        ctx.moe_scores = moe_scores
        ctx.first_a2a = first_a2a
        ctx.second_a2a = second_a2a
        ctx.enter_a2a = enter_a2a
        ctx.a2a_comp = a2a_comp


        mlp_ret = torch.cat(mlp_outputs)
        lay_ret = torch.cat(layernorm_inputs)

        #torch.cuda.nvtx.range_pop()

        #ctx.attention_output = attention_output
        return lay_ret, mlp_ret

    @staticmethod
    def backward(ctx, grad_layernorm_input, grad_mlp_output):
        #if torch.distributed.get_rank() == 0:
        #    print("bwd0")
        attn, self_attention = ctx.attn, ctx.self_attention
        grad_layernorm_inputs = grad_layernorm_input.chunk(attn.chunk_size)
        grad_mlp_outputs = list(grad_mlp_output.chunk(attn.chunk_size))

        batch_size, seqlen, max_s, chunk_len = ctx.batch_size, ctx.seqlen, ctx.max_s, ctx.chunk_len 

        base = 0

        #torch.cuda.nvtx.range_push("BWD Fusion")
        #torch.cuda.synchronize()
        post_events = []
        #torch.cuda.current_stream().wait_stream(get_comp())
        get_comp0().wait_stream(torch.cuda.current_stream())

        for c in range(attn.chunk_size):
            with torch.enable_grad():
                with torch.cuda.stream(get_comp0()):
                    ctx.mlp_outputs[c].backward(grad_mlp_outputs[c])
                    grad_mlp_outputs[c] = None

                    #attn_events.append()

                    post_events.append(torch.cuda.current_stream().record_event())
                    #post_events[-1].record()
        #torch.cuda.synchronize()

        a2a2_events = []
        for c in range(attn.chunk_size):
            with torch.enable_grad():
                with torch.cuda.stream(get_comm()):

                    torch.cuda.current_stream().wait_event(post_events[c])
                    #post_events[c].wait()
                    ctx.second_a2a[0][c].backward(ctx.second_a2a[1][c].grad)
                    del ctx.second_a2a[1][c].grad

                    a2a2_events.append(torch.cuda.current_stream().record_event())
                    #a2a2_events[-1].record()

        comp_events = []
        for c in range(attn.chunk_size):
            with torch.enable_grad():
                with torch.cuda.stream(get_comp0()):
                    #a2a2_events[c].wait()
                    torch.cuda.current_stream().wait_event(a2a2_events[c])
                    
                    ctx.a2a_comp[0][c].backward(ctx.a2a_comp[1][c].grad)
                    del ctx.a2a_comp[1][c].grad

                    comp_events.append(torch.cuda.current_stream().record_event())
                    #comp_events[-1].record()

        #torch.cuda.synchronize()
        a2a1_events = []
        for c in range(attn.chunk_size):
            with torch.enable_grad():
                with torch.cuda.stream(get_comm()):
                    #comp_events[c].wait()

                    torch.cuda.current_stream().wait_event(comp_events[c])

                    ctx.first_a2a[0][c].backward(ctx.first_a2a[1][c].grad)
                    del ctx.first_a2a[1][c].grad

                    a2a1_events.append(torch.cuda.current_stream().record_event())
                    #a2a1_events[-1].record()
        #torch.cuda.synchronize()

        for c in range(attn.chunk_size):
            with torch.enable_grad():
                with torch.cuda.stream(get_comp0()):

                    torch.cuda.current_stream().wait_event(a2a1_events[c])
                    #a2a1_events[c].wait()
                    ctx.enter_a2a[0][c].backward(ctx.enter_a2a[1][c].grad)
                    del ctx.enter_a2a[1][c].grad

                    ctx.moe_scores[1][c].backward(ctx.moe_scores[0][c].grad)
                    del ctx.moe_scores[0][c].grad 
                    ctx.moe_inps[0][c].backward(ctx.moe_inps[1][c].grad + ctx.moe_inps[2][c].grad)
                    del ctx.moe_inps[1][c].grad, ctx.moe_inps[2][c].grad
                
            #with torch.enable_grad():
                    ctx.layernorm_inputs[c].backward(grad_layernorm_inputs[c] + ctx.layernorm_input_xs[c].grad)

        #get_comp().wait_stream(torch.cuda.current_stream())
        #torch.cuda.nvtx.range_pop()
        torch.cuda.current_stream().wait_stream(get_comp0())
        #if torch.distributed.get_rank() == 0:
        #    print("fetch: ", id(ctx.q), id(ctx.k), id(ctx.v))
        #    print("grads: ", ctx.q.grad, ctx.k.grad.norm(), ctx.v.grad.norm(), ctx.q_uses[0].grad.norm())
        q_grads = [rearrange(q.grad, '(b s) ... -> b s ...', b=ctx.batch_size) for q in ctx.q_uses]

        q_grad = torch.cat(q_grads, dim=1).contiguous()
        k_grad = rearrange(ctx.k.grad, '(b s) ... -> b s ...', b=ctx.batch_size).contiguous()
        v_grad = rearrange(ctx.v.grad, '(b s) ... -> b s ...', b=ctx.batch_size).contiguous()
        h_grad = torch.cat([h.grad for h in ctx.hidden_states])

        return q_grad, k_grad, v_grad, h_grad, None, None

def fusionMoE(q, k, v, hidden_states, lst):
    #if len(fusion_g) == 0:
    #    fusion_g.append(torch.cuda.make_graphed_callables(TutelFusionAttMoE.apply, (q, k, v, hidden_states, lst,)))
    #
    #return fusion_g[0](q, k, v, hidden_states, lst,)
    return TutelFusionAttMoE.apply(q, k, v, hidden_states, lst)

class FusionAttMoEModule(torch.nn.Module):
    def __init__(self, use_flash_attn, others):
        super(FusionAttMoEModule, self).__init__()
        self.lst = others
        self.use_flash_attn = use_flash_attn
    def forward(self, q, k, v, hidden_states, attention_mask=None):
        if self.use_flash_attn:
            return TutelFusionAttMoE.apply(q, k, v, hidden_states, self.lst)
        return TutelFusionAttMoENormal.apply(q, k, v, hidden_states, attention_mask, self.lst)
        
class ParallelTransformerLayer(MegatronModule):
    """A single transformer layer.

    Transformer layer takes input with size [s, b, h] and returns an
    output of the same size.
    """

    def __init__(self, init_method, output_layer_init_method,
                 layer_number, layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 drop_path_rate=0.):
        args = get_args()

        super(ParallelTransformerLayer, self).__init__()
        self.layer_number = layer_number
        self.layer_type = layer_type

        self.apply_residual_connection_post_layernorm \
            = args.apply_residual_connection_post_layernorm

        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection

        # Layernorm on the input data.
        self.input_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        # Self attention.
        self.self_attention = ParallelAttention(
            init_method,
            output_layer_init_method,
            layer_number,
            attention_type=AttnType.self_attn,
            attn_mask_type=self_attn_mask_type)
        self.hidden_dropout = args.hidden_dropout
        self.bias_dropout_fusion = args.bias_dropout_fusion
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else None

        # Layernorm on the attention output
        self.post_attention_layernorm = LayerNorm(
            args.hidden_size,
            eps=args.layernorm_epsilon,
            no_persist_layer_norm=args.no_persist_layer_norm,
            sequence_parallel=args.sequence_parallel)

        if self.layer_type == LayerType.decoder:
            self.inter_attention = ParallelAttention(
                init_method,
                output_layer_init_method,
                layer_number,
                attention_type=AttnType.cross_attn)
            # Layernorm on the attention output.
            self.post_inter_attention_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel)

        # MLP
        mlp_cls = ParallelMLP
        if args.moe_num_experts is not None:
            if args.moe_use_megatron_switch:
                mlp_cls = SwitchMLP
            elif args.moe_capacity_factor > 0:
                mlp_cls = MoE
            else:
                mlp_cls = dMoE
        self.mlp = mlp_cls(init_method, output_layer_init_method)

        # Set bias+dropout+add fusion grad_enable execution handler.
        TORCH_MAJOR = int(torch.__version__.split('.')[0])
        TORCH_MINOR = int(torch.__version__.split('.')[1])
        use_nvfuser = TORCH_MAJOR > 1 or (TORCH_MAJOR == 1 and TORCH_MINOR >= 10)
        self.bias_dropout_add_exec_handler = \
                nullcontext if use_nvfuser else torch.enable_grad

        self.fusion_attmoe = args.fusion_attmoe
        self.fusion_split = args.fusion_split 

        self.use_flash_attn = args.use_flash_attn


        self.ampipe = args.ampipe 
        self.pipe_degree = args.pipe_degree

        if self.fusion_attmoe:

            self.FusionAttMoE = FusionAttMoEModule(self.use_flash_attn, [self.self_attention.core_attention_flash if self.use_flash_attn else self.self_attention, self.self_attention, self.hidden_dropout, self.post_attention_layernorm, self.mlp.moe])

            self.usage_attmoe = None 

    def fusion_att_moe_apply(self, layernorm_output, attention_mask, inference_params, hidden_states):
        assert self.apply_residual_connection_post_layernorm == False 
        assert self.drop_path is None 
        assert self.bias_dropout_fusion == True 
        assert self.training == True 
        assert self.layer_type != LayerType.decoder
        assert self.apply_residual_connection_post_layernorm == False 
        assert self.mlp.moe.args.moe_top_k == 1
        assert self.fusion_split != -1

        moe = self.mlp.moe

        q, k, v = self.self_attention(
                    layernorm_output,
                    attention_mask,
                    inference_params=inference_params,
                    skip_core=True)

        chunk_count = self.fusion_split

        bias_dropout_add_func = bias_dropout_add_fused_train

        #print([p.size() for p in hidden_states])
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda
        batch_size, seqlen = q.shape[0], q.shape[1]

        #hidden_states = hidden_states.chunk(chunk_count)

        with tensor_parallel.get_cuda_rng_tracker().fork():
            if not self.use_flash_attn:
                layernorm_inputs, mlp_output = self.FusionAttMoE(q, k, v, hidden_states, attention_mask) #, [attn, self.self_attention, self.hidden_dropout, self.post_attention_layernorm, moe])
            
            else:
            #if self.usage_attmoe is None:
            #    self.usage_attmoe = torch.cuda.make_graphed_callables(self.FusionAttMoE, (q, k, v, hidden_states,))
            #layernorm_inputs, mlp_output = self.usage_attmoe(q, k, v, hidden_states,)
                layernorm_inputs, mlp_output = self.FusionAttMoE(q, k, v, hidden_states) #, [attn, self.self_attention, self.hidden_dropout, self.post_attention_layernorm, moe])

        mlp_bias = moe.bias        

        mlp_output = BWDDEBUG.apply(mlp_output, "MLP")
        
        return mlp_output, mlp_bias, layernorm_inputs


    def fusion_att_moe(self, layernorm_output, attention_mask, inference_params, hidden_states):

        assert self.apply_residual_connection_post_layernorm == False 
        assert self.drop_path is None 
        assert self.bias_dropout_fusion == True 
        assert self.training == True 
        assert self.layer_type != LayerType.decoder
        assert self.apply_residual_connection_post_layernorm == False 
        assert self.mlp.moe.args.moe_top_k == 1
        assert self.fusion_split != -1

        moe = self.mlp.moe
        attn = self.self_attention.core_attention_flash
        q, k, v = self.self_attention(
                    layernorm_output,
                    attention_mask,
                    inference_params=inference_params,
                    skip_core=True)

        chunk_count = self.fusion_split

        bias_dropout_add_func = bias_dropout_add_fused_train
        layernorm_inputs = []
        layernorm_outputs = []
        mlp_outputs = []

        hidden_states = hidden_states.chunk(chunk_count)
        #print([p.size() for p in hidden_states])
        assert q.dtype in [torch.float16, torch.bfloat16]
        assert q.is_cuda
        batch_size, seqlen = q.shape[0], q.shape[1]

        with tensor_parallel.get_cuda_rng_tracker().fork():
            #context_layers = self.self_attention.core_attention_flash(q, k, v)
            k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [k, v]]
            max_s = seqlen
            #context_layers = []
            chunk_len = (seqlen + attn.chunk_size - 1) // attn.chunk_size
            base = 0
            cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                      device=q.device)
            
            for c in range(attn.chunk_size):
                slen = min(seqlen - base, chunk_len)
                cu_seqlens_q = torch.arange(0, (batch_size + 1) * slen, step=slen, dtype=torch.int32,
                                      device=q.device)
                q_use = rearrange(q[:,base:base+slen], 'b s ... -> (b s) ...')

                output_chunk = flash_attn_megablock_call(
                    q_use, k, v, cu_seqlens_q, cu_seqlens, slen, max_s,
                    attn.dropout_p if attn.training else 0.0,
                    softmax_scale=attn.softmax_scale, causal=attn.causal,
                    causal_q_offset=base if attn.causal else 0, #fixed 
                    version=1
                )
                context_layer = rearrange(output_chunk, '(b s) ... -> b s ...', b=batch_size)
                base += slen
            
                context_layer_tmp = rearrange(context_layer, 'b s h d -> s b (h d)').contiguous()
                attention_output, attention_bias = self.self_attention.dense(context_layer_tmp)

                layernorm_input = bias_dropout_add_func(
                        attention_output,
                        attention_bias.expand_as(hidden_states[c]),
                        hidden_states[c],
                        self.hidden_dropout)

                layernorm_inputs.append(layernorm_input)

                layernorm_output = self.post_attention_layernorm(layernorm_input)

                layernorm_outputs.append(layernorm_output)

                mlp_output, _ = self.mlp(layernorm_output)

                mlp_outputs.append(mlp_output)

        layernorm_output = torch.cat(layernorm_outputs)
        layernorm_input = torch.cat(layernorm_inputs)
        mlp_output = torch.cat(mlp_outputs)

        mlp_bias = moe.bias        

        
        return mlp_output, mlp_bias, layernorm_input

    def moe_forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        pass


    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]

        # Layer norm at the beginning of the transformer layer.
        layernorm_output = self.input_layernorm(hidden_states)
        # Self attention.
        if (not self.fusion_attmoe and not self.ampipe) or not self.training:
            #print("normal")
            attention_output, attention_bias = \
                self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params)

            # Residual connection.
            if self.apply_residual_connection_post_layernorm:
                residual = layernorm_output
            else:
                residual = hidden_states

            if self.drop_path is None:
            # jit scripting for a nn.module (with dropout) is not
            # trigerring the fusion kernel. For now, we use two
            # different nn.functional routines to account for varying
            # dropout semantics during training and inference phases.
                if self.bias_dropout_fusion:
                    if self.training:
                        bias_dropout_add_func = bias_dropout_add_fused_train
                    else:
                        bias_dropout_add_func = bias_dropout_add_fused_inference
                else:
                    bias_dropout_add_func = get_bias_dropout_add(self.training)

                with self.bias_dropout_add_exec_handler():
                    layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)
            else:
                out = torch.nn.functional.dropout(attention_output + attention_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
                layernorm_input = residual + self.drop_path(out)

            # Layer norm post the self attention.
            layernorm_output = self.post_attention_layernorm(layernorm_input)

            if self.layer_type == LayerType.decoder:
                attention_output, attention_bias = \
                    self.inter_attention(layernorm_output,
                                     enc_dec_attn_mask,
                                     encoder_output=encoder_output)
                # residual connection
                if self.apply_residual_connection_post_layernorm:
                    residual = layernorm_output
                else:
                    residual = layernorm_input

                with self.bias_dropout_add_exec_handler():
                    layernorm_input = bias_dropout_add_func(
                    attention_output,
                    attention_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Layer norm post the decoder attention
                layernorm_output = self.post_inter_attention_layernorm(layernorm_input)

            # MLP.
            mlp_output, mlp_bias = self.mlp(layernorm_output)
        elif self.ampipe:
            assert self.apply_residual_connection_post_layernorm == False 
            assert self.drop_path is None 
            assert self.bias_dropout_fusion == True 
            assert self.training == True 
            assert self.layer_type != LayerType.decoder
            assert self.apply_residual_connection_post_layernorm == False 
            assert self.mlp.moe.args.moe_top_k == 1

            q, k, v = \
                self.self_attention(
                layernorm_output,
                attention_mask,
                inference_params=inference_params, skip_core=True)
            
            timers = get_timers()
            timer_wrapper_start(timers, 'attn', 0)

            flash = self.self_attention.core_attention_flash
            dense_layer = self.self_attention.dense
            ln = self.post_attention_layernorm
            k, v = [rearrange(x, 'b s ... -> (b s) ...') for x in [k, v]]
            
            out_mlp, layernorm_input = AttMoEPipe.apply(q, k, v, hidden_states, \
                ln.weight, ln.bias, dense_layer.bias,
                    [flash, self.self_attention, dense_layer, \
                        self.pipe_degree, ln, self.hidden_dropout, \
                            self.bias_dropout_add_exec_handler, self.mlp.moe])
            #attention_outputs = torch.chunk(attention_outputs, self.pipe_degree)
            '''
            with tensor_parallel.get_cuda_rng_tracker().fork():
                batch_size, seqlen = q.size(0), q.size(1)
                chunk_size = flash.chunk_size
                context_layers = []
                assert seqlen % chunk_size == 0
                assert flash.causal 
                chunk_len = seqlen // chunk_size
                base = 0
                cu_seqlens = torch.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                      device=q.device)
                cu_seqlens_q = torch.arange(0, (batch_size + 1) * chunk_len, step=chunk_len, dtype=torch.int32,
                                      device=q.device)
                for c in range(chunk_size):
                    q_use = rearrange(q[:,base:base+chunk_len], 'b s ... -> (b s) ...')
                    output_chunk = flash_attn_megablock_call(
                    q_use, k, v, cu_seqlens_q, cu_seqlens, chunk_len, seqlen,
                    flash.dropout_p if flash.training else 0.0,
                    softmax_scale=flash.softmax_scale, causal=True,
                    causal_q_offset=base, #fixed ,
                    version=flash.version
                    )
                    context_layers.append(rearrange(output_chunk, '(b s) h d -> s b (h d)', b=batch_size).contiguous())
                    base += chunk_len
                
                #context_layers = self.self_attention.core_attention_flash(q, k, v) 
            '''
            timer_wrapper_stop(timers, 'attn', 0)
            #context_layers = [rearrange(context_layer, 'b s h d -> s b (h d)').contiguous() for context_layer in context_layers]

            #attention_outputs = []
            #for context_layer in context_layers:
            #    output, attention_bias = self.self_attention.dense(context_layer)
            #    attention_outputs.append(output)

            #residual = hidden_states

            #hidden_states_chunks = hidden_states.chunk(self.pipe_degree, dim=0)

            #layernorm_output = layernorm_output.chunk(self.pipe_degree)

            #outs = []
            #residuals = []
            #for i in range(self.pipe_degree):
            '''
                residual = hidden_states_chunks[i]
                attention_output = attention_outputs[i]
                with self.bias_dropout_add_exec_handler():
                    layernorm_input = bias_dropout_add_fused_train(
                    attention_output,
                    dense_layer.bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

                layernorm_output = self.post_attention_layernorm(layernorm_input)
                '''
            #    mlp_output, mlp_bias = self.mlp(layernorm_output[i])

            #    outs.append(mlp_output)
                #residuals.append(layernorm_input)

            #residual = torch.cat(residuals)
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_fused_train(
                    out_mlp,
                    self.mlp.moe.bias.expand_as(layernorm_input),
                    layernorm_input,
                    self.hidden_dropout)

            output = core.utils.make_viewless_tensor(inp = output,
                                                     requires_grad = output.requires_grad,
                                                     keep_graph = True)
            #outs.append(output)

            return output
        else:
            mlp_output, mlp_bias, layernorm_input = self.fusion_att_moe_apply(layernorm_output,
                    attention_mask,
                    inference_params, hidden_states)
            bias_dropout_add_func = bias_dropout_add_fused_train

        # Second residual connection.
        if self.apply_residual_connection_post_layernorm:
            residual = layernorm_output
        else:
            residual = layernorm_input

        if self.drop_path is None:
            with self.bias_dropout_add_exec_handler():
                output = bias_dropout_add_func(
                    mlp_output,
                    mlp_bias.expand_as(residual),
                    residual,
                    self.hidden_dropout)

            # Jit compiled function creates 'view' tensor. This tensor
            # potentially gets saved in the MPU checkpoint function context,
            # which rejects view tensors. While making a viewless tensor here
            # won't result in memory savings (like the data loader, or
            # p2p_communication), it serves to document the origin of this
            # 'view' tensor.
            output = core.utils.make_viewless_tensor(inp = output,
                                                     requires_grad = output.requires_grad,
                                                     keep_graph = True)

        else:
            out = torch.nn.functional.dropout(mlp_output + mlp_bias,
                                              p=self.hidden_dropout,
                                              training=self.training)
            output = residual + self.drop_path(out)

        return output


class NoopTransformerLayer(MegatronModule):
    """A single 'no-op' transformer layer.

    The sole purpose of this layer is for when a standalone embedding layer
    is used (i.e., args.standalone_embedding_stage == True). In this case,
    zero transformer layers are assigned when pipeline rank == 0. Additionally,
    when virtual pipeline rank >= 1, zero total model parameters are created
    (virtual rank 0 contains the input embedding). This results in the model's
    input and output tensors being the same, which causes an error when
    performing certain memory optimiations on the output tensor (e.g.,
    deallocating it). Thus, this layer disconnects the input from the output
    via a clone. Since ranks containing a no-op layer are generally under-
    utilized (both compute and memory), there's no worry of any performance
    degredation.
    """

    def __init__(self, layer_number):
        super().__init__()
        self.layer_number = layer_number

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        return hidden_states.clone()


def _get_num_layers(args, is_encoder_and_decoder_model, is_decoder=False):
    """Compute the number of transformer layers resident on the current rank."""
    if mpu.get_pipeline_model_parallel_world_size() > 1:
        if is_encoder_and_decoder_model:
            assert args.pipeline_model_parallel_split_rank is not None

            # When a standalone embedding stage is used, a rank is taken from
            # the encoder's ranks, to be used for the encoder's embedding
            # layer. This way, the rank referenced by the 'split rank' remains
            # the same whether or not a standalone embedding stage is used.
            num_ranks_in_encoder = (
                args.pipeline_model_parallel_split_rank - 1
                if args.standalone_embedding_stage else
                args.pipeline_model_parallel_split_rank
            )
            num_ranks_in_decoder = args.transformer_pipeline_model_parallel_size - num_ranks_in_encoder
            assert args.encoder_num_layers % num_ranks_in_encoder == 0, \
                    'encoder_num_layers (%d) must be divisible by number of ranks given to encoder (%d)' % (args.encoder_num_layers, num_ranks_in_encoder)
            assert args.decoder_num_layers % num_ranks_in_decoder == 0, \
                    'decoder_num_layers (%d) must be divisible by number of ranks given to decoder (%d)' % (args.decoder_num_layers, num_ranks_in_decoder)
            if mpu.is_pipeline_stage_before_split():
                num_layers = (
                    0
                    if args.standalone_embedding_stage
                    and mpu.get_pipeline_model_parallel_rank() == 0 else
                    args.encoder_num_layers // num_ranks_in_encoder
                )
            else:
                num_layers = args.decoder_num_layers // num_ranks_in_decoder
        else:
            assert args.num_layers == args.encoder_num_layers
            assert args.num_layers % args.transformer_pipeline_model_parallel_size == 0, \
                'num_layers must be divisible by transformer_pipeline_model_parallel_size'

            # When a standalone embedding stage is used, all transformer layers
            # are divided among pipeline rank >= 1, while on pipeline rank 0,
            # ranks either contain the input embedding layer (virtual pp rank 0),
            # or no layers at all (virtual pp rank >= 1).
            num_layers = (
                0
                if args.standalone_embedding_stage
                and mpu.get_pipeline_model_parallel_rank() == 0 else
                args.num_layers // args.transformer_pipeline_model_parallel_size
            )
    else:
        if not is_decoder:
            num_layers = args.encoder_num_layers
        else:
            num_layers = args.decoder_num_layers
    return num_layers


class ParallelTransformer(MegatronModule):
    """Transformer class."""

    def __init__(self, init_method, output_layer_init_method,
                 layer_type=LayerType.encoder,
                 self_attn_mask_type=AttnMaskType.padding,
                 post_layer_norm=True,
                 pre_process=True, post_process=True,
                 drop_path_rate=0.0):
        super(ParallelTransformer, self).__init__()
        args = get_args()

        self.layer_type = layer_type
        self.model_type = args.model_type
        self.bf16 = args.bf16
        self.fp32_residual_connection = args.fp32_residual_connection
        self.post_layer_norm = post_layer_norm
        self.pre_process = pre_process
        self.post_process = post_process
        self.input_tensor = None
        self.drop_path_rate = drop_path_rate
        self.transformer_impl = args.transformer_impl

        # Store activation checkpoiting flag.
        self.recompute_granularity = args.recompute_granularity
        self.recompute_method = args.recompute_method
        self.recompute_num_layers = args.recompute_num_layers
        self.distribute_saved_activations = \
            args.distribute_saved_activations and not args.sequence_parallel

        self.sequence_parallel = args.sequence_parallel

        # Transformer Engine Init.
        if self.transformer_impl == 'transformer_engine':
            global transformer_engine
            import transformer_engine
        self.use_fp8 = args.fp8_e4m3 or args.fp8_hybrid
        self.fp8_recipe = None
        self.fp8_group = mpu.get_data_parallel_group()
        if self.use_fp8:
            if args.fp8_e4m3:
                fp8_format = transformer_engine.common.recipe.Format.E4M3
            elif args.fp8_hybrid:
                fp8_format = transformer_engine.common.recipe.Format.HYBRID
            self.fp8_recipe = transformer_engine.common.recipe.DelayedScaling(
                margin=args.fp8_margin,
                interval=args.fp8_interval,
                fp8_format=fp8_format,
                amax_history_len=args.fp8_amax_history_len,
                amax_compute_algo=args.fp8_amax_compute_algo,
                override_linear_precision=(False, False, not args.fp8_wgrad),
            )

        self.num_microbatches_in_previous_step = -1
        self.microbatch_count = 0
        self.checkpoint_core_attention = args.recompute_granularity == 'selective'

        # Number of layers.
        self.num_layers = _get_num_layers(
            args,
            args.model_type == ModelType.encoder_and_decoder,
            layer_type == LayerType.decoder)

        self.drop_path_rates = [rate.item() for rate in torch.linspace(0, self.drop_path_rate, args.num_layers)]

        # Transformer layers.
        def build_layer(layer_number):
            if args.transformer_impl == 'local':
                return ParallelTransformerLayer(
                    init_method,
                    output_layer_init_method,
                    layer_number,
                    layer_type=layer_type,
                    self_attn_mask_type=self_attn_mask_type,
                    drop_path_rate=self.drop_path_rates[layer_number - 1])
            else:
                return transformer_engine.pytorch.TransformerLayer(
                    args.hidden_size,
                    args.ffn_hidden_size,
                    args.num_attention_heads,
                    layernorm_epsilon=args.layernorm_epsilon,
                    hidden_dropout=args.hidden_dropout,
                    attention_dropout=args.attention_dropout,
                    init_method=init_method,
                    output_layer_init_method=output_layer_init_method,
                    layer_number=layer_number,
                    kv_channels=args.kv_channels,
                    self_attn_mask_type=self_attn_mask_type.name,
                    tp_group=mpu.get_tensor_model_parallel_group(),
                    get_rng_state_tracker=tensor_parallel.get_cuda_rng_tracker,
                    fuse_wgrad_accumulation=args.gradient_accumulation_fusion,
                    apply_query_key_layer_scaling=args.apply_query_key_layer_scaling,
                    attention_softmax_in_fp32=args.attention_softmax_in_fp32,
                    seq_length=args.seq_length,
                    micro_batch_size=args.micro_batch_size,
                    sequence_parallel=args.sequence_parallel,
                    params_dtype=args.params_dtype,
                    apply_residual_connection_post_layernorm=args.apply_residual_connection_post_layernorm,
                    output_layernorm=False,
                    layer_type="encoder",
                    drop_path_rate=self.drop_path_rates[layer_number - 1],
                    set_parallel_mode=True,
                    fuse_qkv_params=True)

        if args.virtual_pipeline_model_parallel_size is not None:
            assert args.num_layers % args.virtual_pipeline_model_parallel_size == 0, \
                'num_layers_per_stage must be divisible by ' \
                'virtual_pipeline_model_parallel_size'
            assert args.model_type != ModelType.encoder_and_decoder
            # Number of layers in each model chunk is the number of layers in the stage,
            # divided by the number of model chunks in a stage.
            self.num_layers = self.num_layers // args.virtual_pipeline_model_parallel_size
            # With 8 layers, 2 stages, and 4 model chunks, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0]  [2]  [4]  [6]
            # Stage 1: [1]  [3]  [5]  [7]
            # With 8 layers, 2 stages, and 2 virtual stages, we want an assignment of
            # layers to stages like (each list is a model chunk):
            # Stage 0: [0, 1]  [4, 5]
            # Stage 1: [2, 3]  [6, 7]
            offset = mpu.get_virtual_pipeline_model_parallel_rank() * (
                args.num_layers // args.virtual_pipeline_model_parallel_size) + \
                (mpu.get_pipeline_model_parallel_rank() * self.num_layers)
        else:
            # Each stage gets a contiguous set of layers.
            if args.model_type == ModelType.encoder_and_decoder and \
                    mpu.get_pipeline_model_parallel_world_size() > 1:
                pipeline_rank = mpu.get_pipeline_model_parallel_rank()
                if layer_type == LayerType.encoder:
                    offset = pipeline_rank * self.num_layers
                else:
                    num_ranks_in_enc = args.pipeline_model_parallel_split_rank
                    offset = (pipeline_rank - num_ranks_in_enc) * self.num_layers
            else:
                offset = mpu.get_pipeline_model_parallel_rank() * self.num_layers

        if self.num_layers == 0:
            # When a standalone embedding stage is used (e.g.,
            # args.standalone_embedding_stage == True), virtual pipeline ranks
            # on pipeline rank 0 will have zero transformer layers assigned to
            # them. This results in the model's input and output tensors to be
            # the same, which will cause failure for certain output tensor
            # optimizations (e.g., pipeline output deallocation). To remedy
            # this, we assign a 'no-op' layer on these ranks, which will
            # disconnect the input tensor from the output tensor.
            self.num_layers = 1
            self.layers = torch.nn.ModuleList([ NoopTransformerLayer(1) ])
        else:
            self.layers = torch.nn.ModuleList(
                [build_layer(i + 1 + offset) for i in range(self.num_layers)])

        if self.post_process and self.post_layer_norm:
            # Final layer norm before output.
            self.final_layernorm = LayerNorm(
                args.hidden_size,
                eps=args.layernorm_epsilon,
                no_persist_layer_norm=args.no_persist_layer_norm,
                sequence_parallel=args.sequence_parallel)

    def _get_layer(self, layer_number):
        return self.layers[layer_number]

    def _checkpointed_forward(self, hidden_states, attention_mask,
                              encoder_output, enc_dec_attn_mask, is_first_microbatch):
        """Forward method with activation checkpointing."""
        def custom(start, end, is_transformer_engine=False):
            def custom_forward(*args, **kwargs):
                for index in range(start, end):
                    layer = self._get_layer(index)
                    x_ = layer(*args, **kwargs)
                return x_
            def custom_forward_transformer_engine(*args, **kwargs):
                return custom_forward(*args, is_first_microbatch=is_first_microbatch, **kwargs)
            if not is_transformer_engine:
                return custom_forward
            else:
                return custom_forward_transformer_engine

        if self.recompute_method == 'uniform':
            # Uniformly divide the total number of Transformer layers and checkpoint
            # the input activation of each divided chunk.
            # A method to further reduce memory usage reducing checkpoints.
            l = 0
            while l < self.num_layers:
                if self.transformer_impl == 'transformer_engine':
                    hidden_states = transformer_engine.pytorch.distributed.checkpoint(
                        custom(l, l + self.recompute_num_layers, is_transformer_engine=True),
                        self.distribute_saved_activations,
                        tensor_parallel.get_cuda_rng_tracker,
                        mpu.get_tensor_model_parallel_group(),
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                else:
                    hidden_states = tensor_parallel.checkpoint(
                        custom(l, l + self.recompute_num_layers),
                        self.distribute_saved_activations,
                        hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)

                l += self.recompute_num_layers

        elif self.recompute_method == 'block':
            # Checkpoint the input activation of only a set number of individual
            # Transformer layers and skip the rest.
            # A method fully use the device memory removing redundant re-computation.
            for l in range(self.num_layers):
                if l < self.recompute_num_layers:
                    if self.transformer_impl == 'transformer_engine':
                        hidden_states = transformer_engine.pytorch.distributed.checkpoint(
                            custom(l, l + 1, is_transformer_engine=True),
                            self.distribute_saved_activations,
                            tensor_parallel.get_cuda_rng_tracker,
                            mpu.get_tensor_model_parallel_group(),
                            hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                    else:
                        hidden_states = tensor_parallel.checkpoint(
                            custom(l, l + 1),
                            self.distribute_saved_activations,
                            hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                else:
                    if self.transformer_impl == 'transformer_engine':
                        hidden_states = custom(l, l + 1, is_transformer_engine=True)(
                            hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
                    else:
                        hidden_states = custom(l, l + 1)(
                            hidden_states, attention_mask, encoder_output, enc_dec_attn_mask)
        else:
            raise ValueError("Invalid activation recompute method.")

        return hidden_states

    def set_input_tensor(self, input_tensor):
        """Set input tensor to be used instead of forward()'s input.

        When doing pipeline parallelism the input from the previous
        stage comes from communication, not from the input, so the
        model's forward_step_func won't have it. This function is thus
        used by internal code to bypass the input provided by the
        forward_step_func"""
        self.input_tensor = input_tensor

    def forward(self, hidden_states, attention_mask,
                encoder_output=None, enc_dec_attn_mask=None,
                inference_params=None):
        # hidden_states: [s, b, h]

        # Checks.
        if inference_params:
            assert self.recompute_granularity is None, \
                'inference does not work with activation checkpointing'

        if not self.pre_process:
            # See set_input_tensor()
            hidden_states = self.input_tensor

        # Viewless tensor.
        # - We only need to create a viewless tensor in the case of micro batch
        #   size (mbs) == 1, since in this case, 'hidden_states.transpose()'
        #   above creates a view tensor, and '.contiguous()' is a pass-through.
        #   For mbs >= 2, '.contiguous()' creates a new tensor, eliminating
        #   the need to make it viewless.
        #
        #   However, we don't explicitly check mbs == 1 here because
        #   make_viewless_tensor() has negligible overhead when its input
        #   is already viewless.
        #
        # - For the 'else' case above, calling make_viewless_tensor() here is
        #   likely redundant, since p2p_communication.py (likely originator)
        #   already creates viewless tensors. That said, make_viewless_tensor()
        #   is called here to be future-proof and corner-case-proof.
        hidden_states = core.utils.make_viewless_tensor(
            hidden_states,
            requires_grad=True,
            keep_graph=True,
        )

        if self.sequence_parallel:
            rng_context = tensor_parallel.get_cuda_rng_tracker().fork()
        else:
            rng_context = nullcontext()

        with rng_context:
            # The fp8_autocast context manager is a no-op when enabled=True
            # The if...else serves to short circuit name resolution for fp8_autocast
            with transformer_engine.pytorch.fp8_autocast(
                enabled=self.use_fp8,
                fp8_recipe=self.fp8_recipe,
                fp8_group=self.fp8_group
            ) if self.use_fp8 else nullcontext():
                # Determine if the current iteration is first microbatch
                if self.num_microbatches_in_previous_step != get_num_microbatches():
                    self.microbatch_count = 0 # Reset count on new batch size rampup interval
                self.num_microbatches_in_previous_step = get_num_microbatches()
                is_first_microbatch = self.microbatch_count % get_num_microbatches() == 0

                # Forward pass.
                if self.recompute_granularity == 'full':
                    hidden_states = self._checkpointed_forward(hidden_states,
                                                               attention_mask,
                                                               encoder_output,
                                                               enc_dec_attn_mask,
                                                               is_first_microbatch)
                else:
                    forward_kwargs = {
                        'encoder_output': encoder_output,
                        'enc_dec_attn_mask': enc_dec_attn_mask,
                        'inference_params': inference_params,
                    }

                    if self.transformer_impl == 'transformer_engine':
                        forward_kwargs['is_first_microbatch'] = is_first_microbatch
                        forward_kwargs['checkpoint_core_attention'] = self.checkpoint_core_attention

                    for index in range(self.num_layers):
                        layer = self._get_layer(index)
                        timers = get_timers()
                        if mpu.get_pipeline_model_parallel_world_size() == 1:
                            hidden_states = BlockEnter.apply(hidden_states, timers)
                        hidden_states = layer(
                            hidden_states,
                            attention_mask,
                            **forward_kwargs)
                        if mpu.get_pipeline_model_parallel_world_size() == 1:
                            hidden_states = BlockExit.apply(hidden_states, timers)

                # Skip counter update for eval and activation checkpointing
                if torch.is_grad_enabled() and self.training:
                    self.microbatch_count += 1

        # Final layer norm.
        if self.post_process and self.post_layer_norm:
            hidden_states = self.final_layernorm(hidden_states)

        return hidden_states
