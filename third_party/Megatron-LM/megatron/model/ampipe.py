import torch 
from einops import rearrange
from flash_attn.flash_helper import flash_attn_bwd, flash_attn_fwd
from megatron.core import mpu, tensor_parallel, parallel_state
from torch.autograd.function import NestedIOFunction

import os
DEBUG=int(os.environ.get('DEBUG', 1))

class PrepareMoE(torch.autograd.Function):
    def forward(ctx, tokens):
        pass 
    
    def backward(ctx, grad_tokens):
        pass 

class FakeContext():
    def save_for_backward(self, *tensors: torch.Tensor):
        self.to_save = tensors
    @property
    def saved_tensors(self):
        return self.to_save  # type: ignore[misc]


def bias_dropout_add(x, bias, residual, prob, training):
    # type: (Tensor, Tensor, Tensor, float, bool) -> Tensor
    out = torch.nn.functional.dropout(x + bias, p=prob, training=training)
    out = residual + out
    return out

@torch.jit.script
def bias_dropout_add_fused_train(x: torch.Tensor,
                                 bias: torch.Tensor,
                                 residual: torch.Tensor,
                                 prob: float) -> torch.Tensor:
    return bias_dropout_add(x, bias, residual, prob, True)

def bias_dropout_add_ln_fwd(ctx, inp, residual, bias, prob, ln, bias_dropout_add_exec_handler):
    ctx.inp = inp
    ctx.residual = residual
    ctx.bias = bias.detach()
    
    inp.requires_grad = True
    ctx.bias.requires_grad = True 
    residual.requires_grad = True
    with torch.enable_grad():
        ln_input = bias_dropout_add_fused_train(inp, ctx.bias, residual, prob)
    ctx.ln_input = ln_input 
    output = ln.explicit_fwd(ctx, ln_input)
    return output, ln_input

import time
def bias_dropout_add_ln_bwd(ctx, grad_ln_outs, grad_ln_ins, ln):
    grad_fusion, grad_ln_weight, grad_ln_bias = ln.explicit_bwd(ctx, grad_ln_outs)
    with torch.enable_grad():
        ctx.ln_input.backward(grad_fusion + grad_ln_ins)

    return grad_ln_weight, grad_ln_bias, ctx.inp.grad, ctx.residual.grad, ctx.bias.grad

streams = {}
def get_current(dev=None):
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

DISABLEPiPE=int(os.environ.get('DISABLEPiPE', 0))
if DISABLEPiPE:
    get_comp0 = get_current
    get_comm = get_current

class AttMoEPipe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, hidden_states, ln_weight, ln_bias, proj_bias, non_params):
        #torch.cuda.synchronize()
        #t0 = time.time()

        ctx.non_params = non_params
        flash, attn, dense_layer, pipe_degree, ln, hidden_dropout, bias_dropout_add_exec_handler, moe \
            = non_params
        
        ctx.batch_size, seqlen, ctx.head = q.size(0), q.size(1), q.size(2)

        assert seqlen % pipe_degree == 0

        pipe_degree = pipe_degree
        context_layers = []
        chunk_len = seqlen // pipe_degree
        base = 0
        cu_seqlens = torch.arange(0, (ctx.batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                      device=q.device)
        cu_seqlens_q = torch.arange(0, (ctx.batch_size + 1) * chunk_len, step=chunk_len, dtype=torch.int32,
                                      device=q.device)

        ctx.flash_ctx = []
        ctx.dense_ctx = []
        ctx.bdal_ctx = []
        ctx.prepare_ctx = []
        ctx.mlp_ctx = []
        ctx.post_ctx = []
        ctx.loss_ctx = []

        hidden_states_chunks = hidden_states.chunk(pipe_degree, dim=0)

        ln_outs = []
        ln_ins = []
        moe_outs = []
        
        intermediate = [None] * pipe_degree
        dispatchers = [None] * pipe_degree
        scoreses = [None] * pipe_degree
        tokens_per_experts = [None] * pipe_degree


        attn_events = []
        a2a1_events = []
        comp_events = []
        a2a2_events = []

        get_comp0().wait_stream(torch.cuda.current_stream())
        


        for c in range(pipe_degree):
            with torch.cuda.stream(get_comp0()):

                q_use = rearrange(q[:,base:base+chunk_len], 'b s ... -> (b s) ...')
                flash_ctx = FakeContext()
                
                with tensor_parallel.get_cuda_rng_tracker().fork():
                    output_chunk = flash_attn_fwd(flash_ctx,
                    q_use, k, v, cu_seqlens_q, cu_seqlens, chunk_len, seqlen,
                    flash.dropout_p if flash.training else 0.0,
                    softmax_scale=flash.softmax_scale, causal=True,
                    causal_q_offset=base, #fixed ,
                    version=1
                    )
                ctx.flash_ctx.append(flash_ctx)
                context_layers.append(rearrange(output_chunk, '(b s) h d -> s b (h d)', b=ctx.batch_size).contiguous())
                base += chunk_len

                dense_ctx = FakeContext()
                context_layers[-1] = dense_layer.explicit_fwd(dense_ctx, context_layers[-1])
                ctx.dense_ctx.append(dense_ctx)

                #if DEBUG:
                #    tensor_list = [torch.empty_like(context_layers[-1] ) for _ in range(mpu.get_tensor_model_parallel_world_size())]
                #    torch.distributed.all_gather(tensor_list, context_layers[-1] , group=mpu.get_tensor_model_parallel_group())
                #    for t in tensor_list:
                #        assert (t == context_layers[-1] ).all().item(), "not same mlp input across tp ranks"

                bdal_ctx = FakeContext()
                ln_output, ln_input = bias_dropout_add_ln_fwd(bdal_ctx, context_layers[-1], \
                    hidden_states_chunks[c], dense_layer.bias, hidden_dropout, ln, bias_dropout_add_exec_handler)
                ctx.bdal_ctx.append(bdal_ctx)
                ln_ins.append(ln_input)
            #ln_outs.append(ln_output)
                #if DEBUG:
                #    tensor_list = [torch.empty_like(ln_output) for _ in range(mpu.get_tensor_model_parallel_world_size())]
                #    torch.distributed.all_gather(tensor_list, ln_output, group=mpu.get_tensor_model_parallel_group())
                #    for t in tensor_list:
                #        assert (t == ln_output).all().item(), "not same a2a input across tp ranks"


                prepare_ctx = FakeContext()
                a2a_tokens, dispatcher, origin_shape, scores, tokens_per_expert\
                     = moe.tutel_prepare_fwd(prepare_ctx, ln_output)
                ctx.prepare_ctx.append(prepare_ctx)
            
                dispatchers[c] = dispatcher
                intermediate[c] = a2a_tokens
                tokens_per_experts[c] = tokens_per_expert
                scoreses[c] = scores

                attn_events.append(torch.cuda.current_stream().record_event())

        for c in range(pipe_degree):
            with torch.cuda.stream(get_comm()):
                torch.cuda.current_stream().wait_event(attn_events[c])
                #size [4, 256, 512]
                #print("input moe: ", intermediate[c].mean(), torch.distributed.get_rank(), mpu.get_tensor_model_parallel_world_size())
                #if DEBUG:
                #    tensor_list = [torch.empty_like(intermediate[c]) for _ in range(mpu.get_tensor_model_parallel_world_size())]
                #    torch.distributed.all_gather(tensor_list, intermediate[c], group=mpu.get_tensor_model_parallel_group())
                #    for t in tensor_list:
                #        assert (t == intermediate[c]).all().item(), "not same across tp ranks"

                a2a_tokens = moe.tutel_a2a_scatter(intermediate[c], [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])
                intermediate[c] = a2a_tokens

                a2a1_events.append(torch.cuda.current_stream().record_event())
        

        #get_comp0().wait_stream(get_comm())
        #t1 = time.time()
        torch.cuda.synchronize()

        for c in range(pipe_degree):
            with torch.cuda.stream(get_comp0()):
                torch.cuda.current_stream().wait_event(a2a1_events[c])

                mlp_ctx = FakeContext()
                mlp_out = moe.tutel_mlp_fwd(mlp_ctx, intermediate[c])
                ctx.mlp_ctx.append(mlp_ctx)
                intermediate[c] = mlp_out

                comp_events.append(torch.cuda.current_stream().record_event())
            #print("mlp: ", mlp_out.numel() * 4 * 6 * 8)

        for c in range(pipe_degree):
            with torch.cuda.stream(get_comm()):
                torch.cuda.current_stream().wait_event(comp_events[c])

                a2a_tokens = moe.tutel_a2a_gather(intermediate[c], [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])
                intermediate[c] = a2a_tokens

                a2a2_events.append(torch.cuda.current_stream().record_event())


        for c in range(pipe_degree):
            with torch.cuda.stream(get_comp0()):

                torch.cuda.current_stream().wait_event(a2a2_events[c])

                post_ctx = FakeContext()
                post_out = moe.tutel_post_fwd(post_ctx, intermediate[c], dispatchers[c])



                ctx.post_ctx.append(post_ctx)

                post_out = post_out.view(origin_shape)
                moe_outs.append(post_out)

                loss_ctx = FakeContext()
                moe.tutel_loss(loss_ctx, scoreses[c], tokens_per_experts[c])
                ctx.loss_ctx.append(loss_ctx)

            #time.sleep(1000)
        torch.cuda.current_stream().wait_stream(get_comp0())

        
        ret = torch.cat(moe_outs), torch.cat(ln_ins)
        #torch.cuda.synchronize()
        #te = time.time()
        #if torch.distributed.get_rank() == 0:
        #    print("elapsed fwd: ", te - t0, te - t1, t1 - t0)
        return ret 

    @staticmethod
    def backward(ctx, grad_mlp_outs, grad_ln_ins):
        flash, attn, dense_layer, pipe_degree, ln, hidden_dropout, bias_dropout_add_exec_handler, moe\
              = ctx.non_params

        grad_mlp_outs = grad_mlp_outs.chunk(pipe_degree)
        grad_ln_ins = grad_ln_ins.chunk(pipe_degree)
        
        grad_k, grad_v = None, None
        grad_ln_weight, grad_ln_bias, bias_grad = None, None, None
        grad_q = []
        grad_h = []
        intermediate = [None] * pipe_degree
        gates_s_grads = [None] * pipe_degree

        post_events = []
        a2a2_events = []
        comp_events = []
        a2a1_events = []

        get_comp0().wait_stream(torch.cuda.current_stream())

        for c in range(0, pipe_degree):
            with torch.cuda.stream(get_comp0()):
                intermediate[c] = grad_mlp_outs[c].view(-1, grad_mlp_outs[c].size(-1))
        
                intermediate[c], gates_s_grads[c] = moe.tutel_post_bwd(ctx.post_ctx[c], intermediate[c])

                post_events.append(torch.cuda.current_stream().record_event())

        for c in range(0, pipe_degree):
            with torch.cuda.stream(get_comm()):
                torch.cuda.current_stream().wait_event(post_events[c])
                intermediate[c] = moe.tutel_a2a_scatter(intermediate[c], [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])
                a2a2_events.append(torch.cuda.current_stream().record_event())

        torch.cuda.synchronize()

        for c in range(0, pipe_degree):
            with torch.cuda.stream(get_comp0()):
                torch.cuda.current_stream().wait_event(a2a2_events[c])
                intermediate[c] = moe.tutel_mlp_bwd(ctx.mlp_ctx[c], intermediate[c])
                comp_events.append(torch.cuda.current_stream().record_event())
        
        

        for c in range(0, pipe_degree):
            with torch.cuda.stream(get_comm()):
                torch.cuda.current_stream().wait_event(comp_events[c])
                intermediate[c] = moe.tutel_a2a_gather(intermediate[c], [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])
                a2a1_events.append(torch.cuda.current_stream().record_event())

        for c in range(0, pipe_degree):
            with torch.cuda.stream(get_comp0()):
                torch.cuda.current_stream().wait_event(a2a1_events[c])

                
                #if DEBUG:
                #    tensor_list = [torch.empty_like(intermediate[c]) for _ in range(mpu.get_tensor_model_parallel_world_size())]
                #    torch.distributed.all_gather(tensor_list, intermediate[c], group=mpu.get_tensor_model_parallel_group())
                #    for t in tensor_list:
                #        assert (t == intermediate[c]).all().item(), "not same across tp ranks"
                
                grad_ln_out = moe.tutel_prepare_bwd(ctx.prepare_ctx[c], moe.get_loss_grad(ctx.loss_ctx[c]), \
                intermediate[c], gates_s_grads[c]) 

                
                #if DEBUG:
                #    tensor_list = [torch.empty_like(grad_ln_out) for _ in range(mpu.get_tensor_model_parallel_world_size())]
                #    torch.distributed.all_gather(tensor_list, grad_ln_out, group=mpu.get_tensor_model_parallel_group())
                #    for t in tensor_list:
                #        assert (t == grad_ln_out).all().item(), "not same across tp ranks"
                
                #if mpu.get_tensor_model_parallel_world_size() > 1:
                #    torch.distributed.all_reduce(grad_ln_out, op=torch.distributed.ReduceOp.SUM, group=mpu.get_tensor_model_parallel_group())

                d_grad_ln_weight, d_grad_ln_bias, grad_dense, d_hidden_grad, d_bias_grad = \
                bias_dropout_add_ln_bwd(ctx.bdal_ctx[c], grad_ln_out, grad_ln_ins[c], ln)
                grad_h.append(d_hidden_grad)
                grad_ln_weight = grad_ln_weight + d_grad_ln_weight if grad_ln_weight is not None else d_grad_ln_weight
                grad_ln_bias = grad_ln_bias + d_grad_ln_bias if grad_ln_bias is not None else d_grad_ln_bias
                bias_grad = bias_grad + d_bias_grad if bias_grad is not None else d_bias_grad

                feed_flash = dense_layer.explicit_bwd(ctx.dense_ctx[c], grad_dense)

                d_q, d_k, d_v = flash_attn_bwd(ctx.flash_ctx[c], rearrange(feed_flash, 's b (h d) -> (b s) h d', h=ctx.head))
                grad_k = grad_k + d_k if grad_k is not None else d_k
                grad_v = grad_v + d_v if grad_v is not None else d_v  
                grad_q.append(d_q)

        torch.cuda.current_stream().wait_stream(get_comp0())

        return torch.cat([rearrange(gq, '(b s) ... -> b s ...', b=ctx.batch_size) for gq in grad_q], dim=1), \
            grad_k, grad_v, torch.cat(grad_h), grad_ln_weight, grad_ln_bias, bias_grad, None




class XAttMoEPipe(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, hidden_states, ln_weight, ln_bias, proj_bias, non_params):
        ctx.non_params = non_params
        flash, attn, dense_layer, pipe_degree, ln, hidden_dropout, bias_dropout_add_exec_handler, moe \
            = non_params
        
        ctx.batch_size, seqlen, ctx.head = q.size(0), q.size(1), q.size(2)

        assert seqlen % pipe_degree == 0

        pipe_degree = pipe_degree
        context_layers = []
        chunk_len = seqlen // pipe_degree
        base = 0
        cu_seqlens = torch.arange(0, (ctx.batch_size + 1) * seqlen, step=seqlen, dtype=torch.int32,
                                      device=q.device)
        cu_seqlens_q = torch.arange(0, (ctx.batch_size + 1) * chunk_len, step=chunk_len, dtype=torch.int32,
                                      device=q.device)

        ctx.flash_ctx = []
        ctx.dense_ctx = []
        ctx.bdal_ctx = []
        ctx.prepare_ctx = []
        ctx.mlp_ctx = []
        ctx.post_ctx = []
        ctx.loss_ctx = []

        hidden_states_chunks = hidden_states.chunk(pipe_degree, dim=0)

        ln_outs = []
        ln_ins = []
        moe_outs = []
        
        intermediate = [None] * pipe_degree
        dispatchers = [None] * pipe_degree
        scoreses = [None] * pipe_degree
        tokens_per_experts = [None] * pipe_degree
        origin_shapes = [None] * pipe_degree

        attn_events = []
        a2a1_events = []
        comp_events = []
        a2a2_events = []
        


        def pre_comp(c, base):
            q_use = rearrange(q[:,base:base+chunk_len], 'b s ... -> (b s) ...')
            flash_ctx = FakeContext()
            with tensor_parallel.get_cuda_rng_tracker().fork():
                output_chunk = flash_attn_fwd(flash_ctx,
                    q_use, k, v, cu_seqlens_q, cu_seqlens, chunk_len, seqlen,
                    flash.dropout_p if flash.training else 0.0,
                    softmax_scale=flash.softmax_scale, causal=True,
                    causal_q_offset=base, #fixed ,
                    version=1
                    )

            
            ctx.flash_ctx.append(flash_ctx)
            context_layers.append(rearrange(output_chunk, '(b s) h d -> s b (h d)', b=ctx.batch_size).contiguous())

            dense_ctx = FakeContext()
            context_layers[-1] = dense_layer.explicit_fwd(dense_ctx, context_layers[-1])
            ctx.dense_ctx.append(dense_ctx)

            bdal_ctx = FakeContext()
            ln_output, ln_input = bias_dropout_add_ln_fwd(bdal_ctx, context_layers[-1], \
                    hidden_states_chunks[c], dense_layer.bias, hidden_dropout, ln, bias_dropout_add_exec_handler)
            ctx.bdal_ctx.append(bdal_ctx)
            ln_ins.append(ln_input)
            #ln_outs.append(ln_output)

            prepare_ctx = FakeContext()
            a2a_tokens, dispatcher, origin_shapes[c], scores, tokens_per_expert\
                     = moe.tutel_prepare_fwd(prepare_ctx, ln_output)
            ctx.prepare_ctx.append(prepare_ctx)
            
            dispatchers[c] = dispatcher
            intermediate[c] = a2a_tokens
            tokens_per_experts[c] = tokens_per_expert
            scoreses[c] = scores

        def a2a1(c):
            a2a_tokens = moe.tutel_a2a_scatter(intermediate[c], [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])
            intermediate[c] = a2a_tokens

        def mlpx(c):
            mlp_ctx = FakeContext()
            mlp_out = moe.tutel_mlp_fwd(mlp_ctx, intermediate[c])
            ctx.mlp_ctx.append(mlp_ctx)
            intermediate[c] = mlp_out

        def a2a2(c):
            a2a_tokens = moe.tutel_a2a_gather(intermediate[c], [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])
            intermediate[c] = a2a_tokens

        def post_comp(c):
            post_ctx = FakeContext()
            post_out = moe.tutel_post_fwd(post_ctx, intermediate[c], dispatchers[c])
            ctx.post_ctx.append(post_ctx)

            post_out = post_out.view(origin_shapes[c])
            moe_outs.append(post_out)

            loss_ctx = FakeContext()
            moe.tutel_loss(loss_ctx, scoreses[c], tokens_per_experts[c])
            ctx.loss_ctx.append(loss_ctx)

        get_comp0().wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(get_comp0()):
            pre_comp(0, base)
            base += chunk_len                
            attn_events.append(torch.cuda.current_stream().record_event())

        
        for c in range(1, pipe_degree):
            with torch.cuda.stream(get_comp0()):
                pre_comp(c, base)
                base += chunk_len                
                attn_events.append(torch.cuda.current_stream().record_event())

            with torch.cuda.stream(get_comm()):
                torch.cuda.current_stream().wait_event(attn_events[c - 1])
                a2a1(c - 1)
                a2a1_events.append(torch.cuda.current_stream().record_event())


            with torch.cuda.stream(get_comp0()):
                torch.cuda.current_stream().wait_event(a2a1_events[c - 1])
                mlpx(c - 1)
                comp_events.append(torch.cuda.current_stream().record_event())

            with torch.cuda.stream(get_comm()):
                torch.cuda.current_stream().wait_event(comp_events[c - 1])
                a2a2(c - 1)
                a2a2_events.append(torch.cuda.current_stream().record_event())

        for c in range(0, pipe_degree - 1):
            with torch.cuda.stream(get_comp0()):
                torch.cuda.current_stream().wait_event(a2a2_events[c])
                post_comp(c)
        
        with torch.cuda.stream(get_comm()):
            torch.cuda.current_stream().wait_event(attn_events[pipe_degree - 1])
            a2a1(pipe_degree - 1)
            a2a1_events.append(torch.cuda.current_stream().record_event())

        with torch.cuda.stream(get_comp0()):
            torch.cuda.current_stream().wait_event(a2a1_events[pipe_degree - 1])
            mlpx(pipe_degree - 1)
            comp_events.append(torch.cuda.current_stream().record_event())

        with torch.cuda.stream(get_comm()):
            torch.cuda.current_stream().wait_event(comp_events[pipe_degree - 1])
            a2a2(pipe_degree - 1)
            a2a2_events.append(torch.cuda.current_stream().record_event())

        with torch.cuda.stream(get_comp0()):
            torch.cuda.current_stream().wait_event(a2a2_events[pipe_degree - 1])
            post_comp(pipe_degree - 1)
        '''
        for c in range(pipe_degree):
            with torch.cuda.stream(get_comp0()):
                pre_comp(c, base)
                base += chunk_len                
                attn_events.append(torch.cuda.current_stream().record_event())

        for c in range(pipe_degree):
            with torch.cuda.stream(get_comm()):
                torch.cuda.current_stream().wait_event(attn_events[c])
                a2a1(c)
                a2a1_events.append(torch.cuda.current_stream().record_event())
        
        for c in range(pipe_degree):
            with torch.cuda.stream(get_comp0()):
                torch.cuda.current_stream().wait_event(a2a1_events[c])
                mlpx(c)
                comp_events.append(torch.cuda.current_stream().record_event())
            #print("mlp: ", mlp_out.numel() * 4 * 6 * 8)

        for c in range(pipe_degree):
            with torch.cuda.stream(get_comm()):
                torch.cuda.current_stream().wait_event(comp_events[c])

                a2a2(c)
                a2a2_events.append(torch.cuda.current_stream().record_event())

        for c in range(pipe_degree):
            with torch.cuda.stream(get_comp0()):

                torch.cuda.current_stream().wait_event(a2a2_events[c])

                post_comp(c)
        '''
        torch.cuda.current_stream().wait_stream(get_comp0())


        return torch.cat(moe_outs), torch.cat(ln_ins)

    @staticmethod
    def backward(ctx, grad_mlp_outs, grad_ln_ins):
        flash, attn, dense_layer, pipe_degree, ln, hidden_dropout, bias_dropout_add_exec_handler, moe\
              = ctx.non_params

        grad_mlp_outs = grad_mlp_outs.chunk(pipe_degree)
        grad_ln_ins = grad_ln_ins.chunk(pipe_degree)
        
        grad_k, grad_v = None, None
        grad_q = []
        grad_h = []
        
        tokens_grad = grad_mlp_outs[0].view(-1, grad_mlp_outs[0].size(-1))
        tokens_grad, gates_s_grad = moe.tutel_post_bwd(ctx.post_ctx[0], tokens_grad)

        tokens_grad = moe.tutel_a2a_scatter(tokens_grad, [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])

        g_mlp_in = moe.tutel_mlp_bwd(ctx.mlp_ctx[0], tokens_grad)

        tokens_grad = moe.tutel_a2a_gather(g_mlp_in, [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])
        
        grad_ln_out = moe.tutel_prepare_bwd(ctx.prepare_ctx[0], moe.get_loss_grad(ctx.loss_ctx[0]), tokens_grad, gates_s_grad) 


        grad_ln_weight, grad_ln_bias, grad_dense, hidden_grad, bias_grad = \
            bias_dropout_add_ln_bwd(ctx.bdal_ctx[0], grad_ln_out, grad_ln_ins[0], ln)
        grad_h.append(hidden_grad)

        #gradients of proj_weight is directly accumulated
        feed_flash = dense_layer.explicit_bwd(ctx.dense_ctx[0], grad_dense)

        d_q, grad_k, grad_v = flash_attn_bwd(ctx.flash_ctx[0], rearrange(feed_flash, 's b (h d) -> (b s) h d', h=ctx.head))
        grad_q.append(d_q)
        for c in range(1, pipe_degree):

            tokens_grad = grad_mlp_outs[c].view(-1, grad_mlp_outs[c].size(-1))
            
            tokens_grad, gates_s_grad = moe.tutel_post_bwd(ctx.post_ctx[c], tokens_grad)

            tokens_grad = moe.tutel_a2a_scatter(tokens_grad, [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])

            g_mlp_in = moe.tutel_mlp_bwd(ctx.mlp_ctx[c], tokens_grad)

            tokens_grad = moe.tutel_a2a_gather(g_mlp_in, [mpu.get_tensor_model_parallel_world_size(), mpu.get_tensor_model_parallel_group()])
        
            grad_ln_out = moe.tutel_prepare_bwd(ctx.prepare_ctx[c], moe.get_loss_grad(ctx.loss_ctx[c]), \
                tokens_grad, gates_s_grad) 


            d_grad_ln_weight, d_grad_ln_bias, grad_dense, d_hidden_grad, d_bias_grad = \
                bias_dropout_add_ln_bwd(ctx.bdal_ctx[c], grad_ln_out, grad_ln_ins[c], ln)
            grad_h.append(d_hidden_grad)
            grad_ln_weight += d_grad_ln_weight
            grad_ln_bias += d_grad_ln_bias
            bias_grad += d_bias_grad

            feed_flash = dense_layer.explicit_bwd(ctx.dense_ctx[c], grad_dense)

            d_q, d_k, d_v = flash_attn_bwd(ctx.flash_ctx[c], rearrange(feed_flash, 's b (h d) -> (b s) h d', h=ctx.head))
            grad_k += d_k 
            grad_v += d_v 
            grad_q.append(d_q)

        return torch.cat([rearrange(gq, '(b s) ... -> b s ...', b=ctx.batch_size) for gq in grad_q], dim=1), \
            grad_k, grad_v, torch.cat(grad_h), grad_ln_weight, grad_ln_bias, bias_grad, None

