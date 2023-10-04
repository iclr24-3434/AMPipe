import torch
import flash_attn_2_cuda 
import flash_attn_cuda 

def _flash_attn1_forward(q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                        dropout_p, softmax_scale, causal, return_softmax, num_splits=0,
                        generator=None, causal_q_offset=0):
    """
    num_splits: how much to parallelize over the seqlen_q dimension. num_splits=0 means
    it will be set by an internal heuristic. We're exposing num_splits mostly for benchmarking.
    Don't change it unless you know what you're doing.
    """
    softmax_lse, rng_state, *rest = flash_attn_cuda.fwd(
        q, k, v, out, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
        softmax_scale, False, causal, return_softmax, num_splits, causal_q_offset, generator
    )

    S_dmask = rest[0] if return_softmax else None
    return out, softmax_lse, rng_state, S_dmask


def _flash_attn1_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
                         max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
                         rng_state=None, num_splits=0, generator=None, causal_q_offset=0):
    """
    num_splits: whether to parallelize over the seqlen_k dimension (num_splits > 1) or
    not (num_splits = 1). num_splits=0 means it will be set by an internal heuristic.
    Any value above 1 will call the same kernel (i.e. num_splits=2 would call the same kernel
    as num_splits=3), so effectively the choices are 0, 1, and 2.
    This hyperparameter can be tuned for performance, but default value (heuristic) should work fine.
    """
    dout = dout.contiguous()  # CUDA code assumes that dout is contiguous
    _, _, _, softmax_d = flash_attn_cuda.bwd(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal,
        num_splits, causal_q_offset, generator, rng_state)

    return dq, dk, dv, softmax_d



def _flash_attn_varlen_forward(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                               dropout_p, softmax_scale, causal, return_softmax, causal_q_offset):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, q, k, v, out_padded, softmax_lse, S_dmask = flash_attn_2_cuda.varlen_fwd(
        q, k, v, None, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
        softmax_scale, False, causal, return_softmax, causal_q_offset, None
    )

    return out, q, k, v, out_padded, softmax_lse, S_dmask

def _flash_attn_varlen_backward(dout, q, k, v, out, softmax_lse, dq, dk, dv,
                                cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                                dropout_p, softmax_scale, causal, causal_q_offset):
    maybe_contiguous = lambda x: x.contiguous() if x.stride(-1) != 1 else x
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]
    dq, dk, dv, softmax_d, = flash_attn_2_cuda.varlen_bwd(
        dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, False, causal, causal_q_offset, None
    )

    return dq, dk, dv, softmax_d

import time
TIMERS = {}
SELECT = {}
TRYS = {}

class FlashAttnFuncMerge(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p,
                softmax_scale, causal, return_softmax, deterministic, causal_q_offset, version):
        ctx.version = version
        timeit = False
        assert version != 0, "RNG STATE IS DIFFERENT, CAN NOT MIX UP USING VERSION 1 AND VERSION 2"
        if version == 0:
            key = (max_seqlen_q, max_seqlen_k, causal, causal_q_offset, 0) 
            if key in SELECT:
                version = SELECT[key]
            elif key in TRYS:
                tried = TRYS[key]
                for i in [1, 2]:
                    if i not in tried:
                        tried.append(i)
                        version = i 
                        timeit = True
                        break 
                if version == 0:
                    min_id = 0
                    real_time = 10000
                    for i in [1, 2]:
                        if TIMERS[key][i] < real_time:
                            real_time = TIMERS[key][i]
                            min_id = i 
                    SELECT[key] = min_id 
                    version = min_id
            else:
                TRYS[key] = [1]
                version = 1
                timeit = True

        if timeit:
            torch.cuda.synchronize()
            t0 = time.time()

        if version == 1:
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
            out, softmax_lse, _, S_dmask = _flash_attn1_forward(
                q, k, v, torch.empty_like(q), cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
            dropout_p, softmax_scale, causal=causal, return_softmax=return_softmax,
            causal_q_offset=causal_q_offset
            )            
            ctx.save_for_backward(q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state)
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.deterministic = deterministic
            ctx.causal_q_offset = causal_q_offset
            
        elif version == 2:
            ctx.deterministic = deterministic
            rng_state = torch.cuda.get_rng_state() if dropout_p > 0 else None
            ctx.rng_state = rng_state
            if softmax_scale is None:
                softmax_scale = q.shape[-1] ** (-0.5)
            ctx.causal_q_offset = causal_q_offset
            out, q, k, v, out_padded, softmax_lse, S_dmask = _flash_attn_varlen_forward(
                q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                dropout_p, softmax_scale, causal=causal, return_softmax=return_softmax and dropout_p > 0, causal_q_offset=causal_q_offset
            )
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse,
                              cu_seqlens_q, cu_seqlens_k, rng_state)
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
        else:
            assert False

        if timeit:
            torch.cuda.synchronize()
            t1 = time.time()
            if key in TIMERS:
                TIMERS[key][version] = t1 - t0 
            else:
                TIMERS[key] = {version: t1 - t0}
        return out if not return_softmax else (out, softmax_lse, S_dmask)

    @staticmethod
    def backward(ctx, dout, *args):
        version = ctx.version
        timeit = False
        if version == 0:
            key = (ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.causal, ctx.causal_q_offset, 1) 
            if key in SELECT:
                version = SELECT[key]
            elif key in TRYS:
                tried = TRYS[key]
                for i in [1, 2]:
                    if i not in tried:
                        tried.append(i)
                        version = i 
                        timeit = True
                        break 
                if version == 0:
                    min_id = 0
                    real_time = 10000
                    for i in [1, 2]:
                        if TIMERS[key][i] < real_time:
                            real_time = TIMERS[key][i]
                            min_id = i 
                    SELECT[key] = min_id 
                    version = min_id
            else:
                TRYS[key] = [1]
                version = 1
                timeit = True

        if timeit:
            
            torch.cuda.synchronize()
            t0 = time.time()

        if version == 1:
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors

            if rng_state is not None:
                cur_rng_state = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(rng_state)
            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            _flash_attn1_backward(
                dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
                ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p, ctx.softmax_scale, causal= ctx.causal, num_splits=1 if ctx.deterministic else 0,
                causal_q_offset=ctx.causal_q_offset, rng_state=None
            )
            if rng_state is not None:
                torch.cuda.set_rng_state(cur_rng_state)  
        elif version == 2:
            q, k, v, out, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state = ctx.saved_tensors
            
            if rng_state is not None:
                cur_rng_state = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(rng_state)

            dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
            _flash_attn_varlen_backward(
                dout, q, k, v, out, softmax_lse, dq, dk, dv, cu_seqlens_q, cu_seqlens_k,
                ctx.max_seqlen_q, ctx.max_seqlen_k, ctx.dropout_p, ctx.softmax_scale, ctx.causal, causal_q_offset=ctx.causal_q_offset
            )
            dq = dq[..., :dout.shape[-1]]  # We could have padded the head dimension
            dk = dk[..., :dout.shape[-1]]
            dv = dv[..., :dout.shape[-1]]
            if rng_state is not None:
                torch.cuda.set_rng_state(cur_rng_state)            
        else:
            assert False


        if timeit:
            torch.cuda.synchronize()
            t1 = time.time()
            if key in TIMERS:
                TIMERS[key][version] = t1 - t0 
            else:
                TIMERS[key] = {version: t1 - t0}
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None
        





def flash_attn_megablock_call(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                           dropout_p=0.0, softmax_scale=None, causal=False,
                           return_attn_probs=False, deterministic=False, causal_q_offset=0, version=1):

    return FlashAttnFuncMerge.apply(
        q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale, causal, return_attn_probs, deterministic, causal_q_offset, version
    )

def flash_attn_fwd(ctx, q_use, k, v, cu_seqlens_q, cu_seqlens, chunk_len, seqlen, dropout_p, softmax_scale, 
    causal, causal_q_offset, version):
    assert causal and version == 1
    out = FlashAttnFuncMerge.forward(ctx, q_use, k, v, cu_seqlens_q, cu_seqlens, chunk_len, seqlen, 
    dropout_p, softmax_scale, True, False, False, causal_q_offset, 1
    )
    return out 


def flash_attn_bwd(ctx, dout):
    ret = FlashAttnFuncMerge.backward(ctx, dout)
    return ret[0], ret[1], ret[2] 
