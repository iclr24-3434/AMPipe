from megablocks.layers import common
from megablocks.layers import mpu
from megablocks.layers import router
from megablocks.layers import mlp
from megablocks.layers.all_to_all import all_to_all
from megablocks.layers.arguments import Arguments
import megablocks.ops as ops
import numpy as np
import torch
import tutel
import megablocks_ops
import tutel.impls.communicate as C
import tutel_custom_kernel

class BWDDEBUG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, info):
        ctx.info = info 
        return inp

    @staticmethod
    def backward(ctx, grad_inp):
        if torch.distributed.get_rank() == 0:
            print("CALLING BWD: ", ctx.info)
        return grad_inp, None 



def create_fake(x):
    return megablocks_ops.fake_tensor(x)
    return x.detach() #if last line reports error

class NoBuffer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        fake = create_fake(x) #watch out, do not access to x's data_ptr
        return fake
    @staticmethod
    def backward(ctx, g):
        return g

class NoBufferAssist(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, assistant):
        #x.size() == assistant
        assert x.size() == assistant.size()
        assert assistant.requires_grad == False
        return assistant
    @staticmethod
    def backward(ctx, g):
        return g, None

class MLP_TP_F(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, group):
        ctx.group = group 
        return tokens
    @staticmethod
    def backward(ctx, g_tokens):
        torch.distributed.all_reduce(g_tokens, op=torch.distributed.ReduceOp.SUM, group=ctx.group)
        return g_tokens, None

class MLP_TP_G(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens, group):
        torch.distributed.all_reduce(tokens, op=torch.distributed.ReduceOp.SUM, group=group)
        return tokens
    @staticmethod
    def backward(ctx, g_tokens):
        return g_tokens, None

_LOAD_BALANCING_LOSS = []

_MoE_Layer = []

import os
FAKE_A2A_SCALE=int(os.environ.get('FAKE_A2A_SCALE', 1))

def save_load_balancing_loss(loss, idx=-1):
    global _LOAD_BALANCING_LOSS
    if idx == -1 or len(_LOAD_BALANCING_LOSS) <= idx:
        _LOAD_BALANCING_LOSS.append(loss)
    else:
        _LOAD_BALANCING_LOSS[idx] = (_LOAD_BALANCING_LOSS[idx][0] + loss[0], torch.cat([_LOAD_BALANCING_LOSS[idx][1], loss[1]]))

def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()

def get_world_size(group=None):
    try:
        return torch.distributed.get_world_size(group)
    except:
        return 1

def batched_load_balancing_loss(args : Arguments):
    # tokens_per_expert[i].shape = (num_experts)
    # expert_scores[i].shape = (tokens, num_experts)
    tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())
    num_layers_per_pipeline_stage = (
        args.num_layers // args.pipeline_model_parallel_size)
    if args.num_layers_per_virtual_pipeline_stage is not None:
        num_layers_per_pipeline_stage = args.num_layers_per_virtual_pipeline_stage

    if len(tokens_per_expert) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} token_per_experts "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{args.num_layers}\npipeline_model_parallel_size = "
            f"{args.pipeline_model_parallel_size}\n"
            "num_layers_per_virtual_pipeline_stage"
            f" = {args.num_layers_per_virtual_pipeline_stage}")
    if len(expert_scores) != num_layers_per_pipeline_stage:
        raise ValueError(
            f"Expected {num_layers_per_pipeline_stage} expert_scores "
            f"but found {len(tokens_per_expert)}.\nnum_layers = "
            f"{args.num_layers}\npipeline_model_parallel_size = "
            f"{args.pipeline_model_parallel_size}\n"
            "num_layers_per_virtual_pipeline_stage"
            f" = {args.num_layers_per_virtual_pipeline_stage}")

    # Verify the shape of the tokens_per_expert and expert_scores tensors.
    assert all([
        x.ndim == 1 and x.numel() == args.moe_num_experts
        for x in tokens_per_expert
    ])

    tokens = expert_scores[0].shape[0]
    assert all([
        (x.ndim == 2 and x.shape[1] == args.moe_num_experts and
         x.shape[0] == tokens) for x in expert_scores
    ])


    # Concatenate the contributions of each layer and convert to
    # the correct types and formats for the dot product.
    if args.moe_lbl_in_fp32:
        expert_scores = torch.cat(expert_scores, dim=1).float().mean(dim=0)
    else:
        expert_scores = torch.cat(expert_scores, dim=1).mean(dim=0)
    tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype)

    expected_values = num_layers_per_pipeline_stage * args.moe_num_experts
    assert tokens_per_expert.numel() == expected_values
    assert expert_scores.numel() == expected_values

    # Calculate the total scale across all factors.
    #
    # loss_weight * num_experts / (num_layers * tokens * top_k)
    scale_numerator = (
        args.moe_num_experts *
        args.moe_loss_weight
    )
    scale_denominator = (
        args.num_layers *
        tokens *
        args.moe_top_k
    )
    scale = scale_numerator / scale_denominator
    return scale * torch.dot(tokens_per_expert, expert_scores)


class MoE(torch.nn.Module):

    def __init__(self, args : Arguments):
        super(MoE, self).__init__()
        self.args = args

        # Calculate the number of experts in total and the number of experts
        # owned by this rank.
        world_size = mpu.get_expert_parallel_world_size(args)
        self.num_experts = args.moe_num_experts
        self.top_k = self.args.moe_top_k

        # Calculate the number of bits needed to represent the expert indices
        # so that we can pass it to radix sort.
        self.sort_end_bit = max(int(np.ceil(np.log2(self.num_experts))), 1)

        # Token router.
        self.router = router.LearnedRouter(args)

        # Expert MLP.
        self.mlp = mlp.MLP(args)

        # Note that the output bias is not parallelized with expert
        # model parallelism.
        self.bias = torch.nn.Parameter(torch.empty(
            1, 1, args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        torch.nn.init.zeros_(self.bias)

        # Select the forward function for the operating mode.
        self.forward_fn = (
            self.parallel_forward_once if
            args.moe_expert_model_parallelism else
            self.forward_once)
        
        global _MoE_Layer
        self.moe_id = len(_MoE_Layer)
        _MoE_Layer.append(self.moe_id)

    def expert_capacity(self, tokens):
        world_size = mpu.get_expert_parallel_world_size(self.args)
        tokens_per_expert = (
            self.top_k * tokens * world_size / self.num_experts)
        return int(self.args.moe_capacity_factor * tokens_per_expert)

    def load_balancing_loss(self, tokens_per_expert, expert_scores):
        """Calculate the load balancing loss contribution."""
        assert len(expert_scores.size()) == 2
        tokens, num_experts = expert_scores.size()
        assert num_experts == self.num_experts
        assert len(tokens_per_expert.size()) == 1
        num_experts, = tokens_per_expert.size()
        assert num_experts == self.num_experts
        scale = self.num_experts / (tokens * self.top_k)
        return scale * torch.dot(
            tokens_per_expert.to(expert_scores.dtype),
            expert_scores.mean(dim=0))

    def indices_and_bins(self, top_expert):
        # Sort the expert ids to produce the scatter/gather
        # indices for the permutation.
        #
        # TODO(tgale): Is it worth doing this conversion to 32-bit
        # prior? Could we place the `torch.max` operation to return
        # 32-bit expert indices?
        top_expert = top_expert.int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)

        # Histogram the expert ids to identify the number of
        # tokens routed to each expert.
        #
        # TODO(tgale): Does the sorted data produce a more favorable
        # data distribution for histogram? Or is the op parallelism
        # worth more?
        tokens_per_expert = ops.histogram(top_expert, self.num_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    def permute_and_compute(
            self,
            x,
            tokens_per_expert, # unused
            indices,
            bin_ids, # unused
            expert_weights,
            bins,
            expert_capacity,
            top_k):
        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        x = ops.binned_gather(
            x, indices, bins, expert_capacity, top_k)

        # Perform the expert computation. Note that we don't
        # use biases for these linear operations.
        x = self.mlp(x)

        # Un-route the data for the MoE output.
        return ops.binned_scatter(
            x, indices, expert_weights, bins, top_k)

    def forward_once(self, x, expert_weights, top_experts):
        # x: [sl, bs, hs]
        # expert_weights: [sl * bs, top-k]
        # top_experts: [sl * bs, top-k]
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_experts))

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            sl, bs, hs = x.size()
            expert_capacity = self.expert_capacity(sl * bs)
            if expert_capacity == 0:
                expert_capacity = torch.max(tokens_per_expert).item()

        x = self.permute_and_compute(
            x,
            tokens_per_expert,
            indices,
            bin_ids,
            expert_weights,
            bins,
            expert_capacity,
            self.top_k)
        return x, tokens_per_expert

    def parallel_forward_once(self, x, expert_weights, top_experts):
        # NOTE: This function implements the same computation as forward_once
        # but with expert model parallelism.
        #
        # 1. Permute the tokens locally so that they are grouped by their
        # expert assignments. This allows us to transfer all of the tokens
        # for a remote device in one communication primitive.
        #
        # 2. Permute the tokens across the expert parallel devices. After
        # this is completed each device has all of the tokens assigned to
        # its set of experts in its local HBM.
        #
        # 3. Permute the tokens locally so that they are grouped by their
        # expert assignement. After the distributed permutation the tokens
        # are grouped by which device they came from. We re-order them
        # locally to allow for efficient computation.
        #
        # After this series of permutations we compute the linear layers
        # and then repeat these three steps in reverse to produce the final
        # output.
        #
        # Compute the mapping of local tokens to experts.
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_experts))

            # If we're sharding the experts along the hidden dimension
            # multiple devices own parts of the same sets of experts.
            # Replicate the token counts so every device gets the counts.
            repeated_tokens_per_expert = tokens_per_expert.repeat(
                mpu.hidden_sharding_degree(self.args))

            # Pass token count information to the device on which the
            # target expert resides.
            parallel_tokens_per_expert = torch.empty_like(repeated_tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert,
                repeated_tokens_per_expert,
                group=self.args.expert_parallel_group,
                async_op=True)

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        #
        # TODO(tgale): We can tune these kernels for this special case by
        # skipping the memset if tokens == padded_tokens and also taking
        # in an optional padded_tokens rather than copying it from the
        # device.
        #
        # This view updates the shape of the tensor from [sl, bs, hs] to
        # [sl * bs, hs] prior to the permutation.
        x = x.view(-1, x.shape[-1])
        x = ops.padded_gather(
            x,
            indices,
            bin_ids,
            bins,
            bins,
            self.top_k)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()
            experts_per_rank = mpu.experts_per_rank(self.args)

            # Reshape to [world_size, num_experts_per_rank].
            world_size = mpu.get_expert_parallel_world_size(self.args)
            repeated_tokens_per_expert = (
                repeated_tokens_per_expert.view(world_size, experts_per_rank))
            parallel_tokens_per_expert = (
                parallel_tokens_per_expert.view(world_size, experts_per_rank))

            # TODO(tgale): It might be faster to do this on the GPU and
            # then communicate the results back to the host.
            send_counts = repeated_tokens_per_expert.cpu().sum(dim=-1)
            recv_counts = parallel_tokens_per_expert.cpu().sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

        # If we're sharding the experts along the hidden dimension
        # multiple devices own parts of the same sets of experts.
        # Replicate the token counts so devices that share experts
        # get all of the tokens assigned to them.
        #
        # TODO(tgale): Fuse this into the prior, local permutation.
        x = x.repeat(mpu.hidden_sharding_degree(self.args), 1)

        # Start the cross-device permutation asynchronously so we can
        # overlap communication with computation.
        parallel_x, parallel_x_handle = all_to_all(
            x, recv_counts, send_counts,
            self.args.expert_parallel_group,
            async_op=True)

        # Reduce along the hidden sharding to get the final outputs.
        #
        # TODO(tgale): Fuse this into the following local permutation.

        with torch.no_grad():
            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.
            replicate_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert.flatten(), 0)
            replicate_bins = (
                replicate_bins.view(1)
                if not len(replicate_bins.size())
                else replicate_bins
            )

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(
                    self.num_experts * mpu.hidden_sharding_degree(self.args),
                    dtype=torch.int32,
                    device=indices.device
                ),
                mpu.experts_per_rank(self.args),
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0),
                replicate_bins, tokens_received).flatten()

            # TODO(tgale): The sort_end_bit here can be reduced.
            parallel_bin_ids, parallel_indices = ops.sort(
                parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(
                dim=0, dtype=torch.int)
            parallel_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert, 0)
            parallel_bins = (
                parallel_bins.view(1)
                if not len(parallel_bins.size())
                else parallel_bins
            )

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, hs = x.size()
            expert_capacity = self.expert_capacity(tokens)
            if expert_capacity == 0:
                expert_capacity = torch.max(
                    parallel_tokens_per_expert).item()

        # Locally permute the tokens and perform the expert computation.
        # Block to make sure that the cross-device permutation is complete.
        parallel_x_handle.wait()
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            None,  # expert_weights
            parallel_bins,
            expert_capacity,
            top_k=1)

        # Un-permute the tokens across the devices.
        x, _ = all_to_all(
            parallel_x, send_counts, recv_counts,
            self.args.expert_parallel_group)

        shape = (
            mpu.hidden_sharding_degree(self.args),
            -1,
            self.args.hidden_size
        )
        x = x.view(shape).sum(dim=0)
        
        # Un-permute locally to setup for the next series of operations.
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            bins,
            self.top_k)
        return x, tokens_per_expert.flatten()


    def parallel_forward_prepare(self, x, top_expert):
        # NOTE: This function implements the same computation as forward_once
        # but with expert model parallelism.
        #
        # 1. Permute the tokens locally so that they are grouped by their
        # expert assignments. This allows us to transfer all of the tokens
        # for a remote device in one communication primitive.
        #
        # 2. Permute the tokens across the expert parallel devices. After
        # this is completed each device has all of the tokens assigned to
        # its set of experts in its local HBM.
        #
        # 3. Permute the tokens locally so that they are grouped by their
        # expert assignement. After the distributed permutation the tokens
        # are grouped by which device they came from. We re-order them
        # locally to allow for efficient computation.
        #
        # After this series of permutations we compute the linear layers
        # and then repeat these three steps in reverse to produce the final
        # output.
        #
        # Compute the mapping of local tokens to experts.
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_expert))

            # Pass token count information to the device on which the
            # target expert resides.
            parallel_tokens_per_expert = torch.empty_like(
                tokens_per_expert)
            tpe_handle = torch.distributed.all_to_all_single(
                parallel_tokens_per_expert,
                tokens_per_expert,
                group=self.args.expert_parallel_group,
                async_op=True)

        # Permute locally and without any padding so that tokens for each
        # parallel device are stored contiguously.
        #
        # TODO(tgale): We can tune these kernels for this special case by
        # skipping the memset if tokens == padded_tokens and also taking
        # in an optional padded_tokens rather than copying it from the
        # device.
        #
        # This view updates the shape of the tensor from [sl, bs, hs] to
        # [sl * bs, hs] prior to the permutation.
        x = x.view(-1, x.shape[-1])
        x = ops.padded_gather(x, indices, bin_ids, bins, bins)

        # Compute the number of tokens that will be received from each
        # device and permute the input data across the devices.
        with torch.no_grad():
            tpe_handle.wait()
            world_size = mpu.get_expert_parallel_world_size(self.args)

            # Reshape to [world_size, num_experts_per_rank].
            tokens_per_expert = tokens_per_expert.view(world_size, -1)
            parallel_tokens_per_expert = (
                parallel_tokens_per_expert.view(world_size, -1))

            # TODO(tgale): It might be faster to do this on the GPU and
            # then communicate the results back to the host.
            send_counts = tokens_per_expert.cpu().sum(dim=-1)
            recv_counts = parallel_tokens_per_expert.cpu().sum(dim=-1)

            # Convert the send/recv counts to lists.
            send_counts = send_counts.tolist()
            recv_counts = recv_counts.tolist()
            tokens_received = sum(recv_counts)

            # After we do the cross-device permutation we have the tokens on the
            # correct device but not yet grouped by expert because we received
            # tokens from each device as contiguous chunks. To group the tokens
            # for expert computation we'll do one more local permutation. The
            # rest of this torch.no_grad() scope sets up the indices and bins
            # for this permutation.
            replicate_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert.flatten(), 0)
            replicate_bins = (
                replicate_bins.view(1)
                if not len(replicate_bins.size())
                else replicate_bins
            )

            # Construct the expert indices for the permuted tokens.
            parallel_top_expert = torch.remainder(
                torch.arange(
                    self.num_experts, dtype=torch.int32, device=indices.device),
                self.num_experts_per_rank,
            )
            parallel_top_expert = ops.replicate(
                parallel_top_expert.unsqueeze(dim=0),
                replicate_bins, tokens_received).flatten()

            # TODO(tgale): The sort_end_bit here can be reduced.
            parallel_bin_ids, parallel_indices = ops.sort(
                parallel_top_expert, self.sort_end_bit)

            # Calculate the bins boundaries from the token counts.
            parallel_tokens_per_expert = parallel_tokens_per_expert.sum(
                dim=0, dtype=torch.int)
            parallel_bins = ops.inclusive_cumsum(
                parallel_tokens_per_expert, 0)
            parallel_bins = (
                parallel_bins.view(1)
                if not len(parallel_bins.size())
                else parallel_bins
            )

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            tokens, hs = x.size()
            expert_capacity = self.expert_capacity(tokens)
            if expert_capacity == 0:
                expert_capacity = torch.max(parallel_tokens_per_expert)
        return x, recv_counts, send_counts, parallel_tokens_per_expert, \
            parallel_indices, parallel_bin_ids, parallel_bins, expert_capacity, \
            indices, bin_ids, bins, tokens_per_expert

    def parallel_forward_a2a1(self, x, recv_counts, send_counts):
        # Permute the tokens across the devices.
        parallel_x = all_to_all(
            x, recv_counts, send_counts,
            self.args.expert_parallel_group)
        return parallel_x
        # Locally permute the tokens and perform the expert computation.
    def parallel_forward_compute(self, parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            expert_capacity):
        parallel_x = self.permute_and_compute(
            parallel_x,
            parallel_tokens_per_expert,
            parallel_indices,
            parallel_bin_ids,
            parallel_bins,
            expert_capacity)
        return parallel_x

    def parallel_forward_a2a2(self, parallel_x, send_counts, recv_counts):
        # Un-permute the tokens across the devices.
        x = all_to_all(
            parallel_x, send_counts, recv_counts,
            self.args.expert_parallel_group)
        return x 

    def parallel_forward_post(self, x, indices, bin_ids, bins, tokens_per_expert):

        # Un-permute locally to setup for the next series of operations.
        x = ops.padded_scatter(
            x,
            indices,
            bin_ids,
            expert_weights,
            bins,
            bins,
            self.top_k)
        return x, tokens_per_expert.flatten()
    
    def tutel_prepare(self, x, scores):
        origin_shape = x.shape 
        x = x.view(-1, origin_shape[-1])
        crit, top_experts = tutel.tutel_moe.extract_critical(scores,
                top_k = self.args.moe_top_k,
                loss_fn = None,
                capacity_factor = self.args.moe_capacity_factor
            )

        tokens_per_expert = ops.histogram(top_experts.view(-1), self.num_experts)

        y = tutel.tutel_moe.fast_encode(x.to(scores.dtype), crit, True).to(x.dtype)
        return y, tokens_per_expert, crit

    def tutel_a2a1(self, x):
        return tutel.impls.communicate.all_to_all(x, 1, 0, use_2dh=False, group=self.args.expert_parallel_group)
    
    def tutel_a2a2(self, x):
        return tutel.impls.communicate.all_to_all(x, 0, 1, use_2dh=False, group=self.args.expert_parallel_group)

    def tutel_post(self, x, crit, dtype):
        y = tutel.tutel_moe.fast_decode(x.to(dtype), crit, True)
        return y

    def hash_forward(self, x, timers, start, stop):
        pass 

    def tutel_prepare_fwd(self, ctx, x):
        ctx.x0 = x.detach()
        ctx.x0.requires_grad = True
        with torch.enable_grad():
            scores = self.router.tutel_forward(ctx.x0)
        ctx.scores = scores 
        origin_shape = x.shape 
        x = x.view(-1, origin_shape[-1])

        y, tokens_per_expert, dispatcher = tutel.impls.fast_dispatch.extract_critical_encode(ctx, x, scores,
                top_k = self.args.moe_top_k,
                loss_fn = None,
                capacity_factor = self.args.moe_capacity_factor
            )

        return y, dispatcher, origin_shape, scores, tokens_per_expert 
        #y, crit, dispatcher = tutel.tutel_moe.fast_encode(x.to(scores.dtype), crit, True).to(x.dtype)


    def tutel_prepare_bwd(self, ctx, g_score, g_tokens, g_gates):
        
        grad_x = tutel.impls.fast_dispatch.encode_bwd(ctx, g_tokens)
        for g_gate, gate in zip(g_gates, ctx.gates_s):
            gate.backward(g_gate)

        #print("score0:", ctx.scores0.grad)
        ctx.scores.backward(g_score + ctx.scores0.grad)
        #print("bwd: ", g_tokens.size(), grad_x.size(), ctx.x0.size(), ctx.x0.grad.size())
        grad_x = grad_x.view(ctx.x0.grad.size())
        return grad_x + ctx.x0.grad

    def tutel_mlp_fwd(self, ctx, tokens):
        ctx.tokens = tokens.detach()
        ctx.tokens.requires_grad = True
        with torch.enable_grad():
            y = self.mlp(ctx.tokens)
            ctx.y = NoBuffer.apply(y)
        return y 

    def tutel_mlp_bwd(self, ctx, g_tokens):
        ctx.y.backward(g_tokens)
        return ctx.tokens.grad

    def tutel_a2a_scatter(self, tokens, tp_info):
        group = self.args.expert_parallel_group
        world_size = get_world_size(group) #world size not include TP ranks
        if world_size == 1:
            return tokens 
        
        tokens = tokens.contiguous()
        output = torch.empty_like(tokens)

        C.AllToAllStatus.init(group, -1, -1)
        tutel_custom_kernel.all_to_all_with_scale(tokens, output, FAKE_A2A_SCALE)
        '''
        torch.distributed.all_to_all_single(output, tokens, group=group)
        if FAKE_A2A_SCALE > 1:
            for i in range(FAKE_A2A_SCALE - 1):
                torch.distributed.all_to_all_single(output, tokens, group=group)
        '''


        output = output.view([world_size, -1] + list(output.shape[1:]))
        output = output.permute([1, 0] + list(range(2, output.dim())))
        #print("o0.size: ", output.size()) #torch.Size([1, 8, 1280, 512])
        output = output.contiguous().view(list(output.shape[:1]) + [-1] + list(output.shape[3:]))
        #[1, 10240, 512]
        #y = tutel.impls.communicate.all_to_all(y, 1, 0, use_2dh=False, group=self.args.expert_parallel_group)
        return output 
    
    def tutel_a2a_scatter_p0(self, tokens):
        world_size = get_world_size(self.args.expert_parallel_group)
        if world_size == 1:
            return tokens 
        tokens = tokens.contiguous()
        output = torch.empty_like(tokens)
        return tokens, output 
    
    def tutel_a2a_scatter_p1(self, tokens, output):
        C.AllToAllStatus.init(self.args.expert_parallel_group, -1, -1)
        tutel_custom_kernel.all_to_all_with_scale(tokens, output, FAKE_A2A_SCALE)
    
    def tutel_a2a_scatter_p2(self, output):
        output = output.view([world_size, -1] + list(output.shape[1:]))
        output = output.permute([1, 0] + list(range(2, output.dim())))
        #print("o0.size: ", output.size()) #torch.Size([1, 8, 1280, 512])
        output = output.contiguous().view(list(output.shape[:1]) + [-1] + list(output.shape[3:]))
        return output 

    def tutel_a2a_gather(self, tokens, tp_info):
        group = self.args.expert_parallel_group
        world_size = get_world_size(group)
        if world_size == 1:
            return tokens 



        reshaped_input = tokens.view(list(tokens.shape[:1]) + [world_size, -1] + list(tokens.shape[2:]))
        reshaped_input = reshaped_input.permute([1, 0] + list(range(2, reshaped_input.dim()))).contiguous()
        #simple_all_to_all(reshaped_input, group, background=True)
        local_input = torch.empty_like(reshaped_input)

        C.AllToAllStatus.init(group, -1, -1)
        tutel_custom_kernel.all_to_all_with_scale(reshaped_input, local_input, FAKE_A2A_SCALE)


        '''        
        torch.distributed.all_to_all_single(local_input, reshaped_input, group=group)

        if FAKE_A2A_SCALE > 1:
            for i in range(FAKE_A2A_SCALE - 1):
                torch.distributed.all_to_all_single(local_input, reshaped_input, group=group)
        '''     
        local_input = local_input.view([-1] + list(local_input.shape[2:]))

        if tp_info[0] > 1 :
            torch.distributed.all_reduce(local_input, op=torch.distributed.ReduceOp.SUM, group=tp_info[1])

        return local_input 
    
    def tutel_post_fwd(self, ctx, tokens, dispatcher):
        
        tokens = tutel.impls.fast_dispatch.decode_fwd(ctx, tokens, dispatcher)

        return tokens

    def tutel_post_bwd(self, ctx, g_tokens):
        tokens_grad, scores_grad = tutel.impls.fast_dispatch.decode_bwd(ctx, g_tokens)
        return tokens_grad, scores_grad

    def tutel_loss(self, ctx, scores, tokens_per_expert):
        ctx.scores = scores.detach()
        ctx.scores.requires_grad = True
        with torch.enable_grad():
            save_load_balancing_loss((tokens_per_expert, ctx.scores), self.moe_id)
    
    def get_loss_grad(self, ctx):
        ret = ctx.scores.grad 
        del ctx.scores 
        return ret 

    def tutel_forward(self, x, timers, start, stop, tp_info):
        scores = self.router.tutel_forward(x)
        origin_shape = x.shape 
        x = x.view(-1, origin_shape[-1])
        crit, top_experts = tutel.tutel_moe.extract_critical(scores,
                top_k = self.args.moe_top_k,
                loss_fn = None,
                capacity_factor = self.args.moe_capacity_factor
            )

        tokens_per_expert = ops.histogram(top_experts.view(-1), self.num_experts)

        y = tutel.tutel_moe.fast_encode(x.to(scores.dtype), crit, True).to(x.dtype)

        if tp_info[0] > 1:
            y = MLP_TP_F.apply(y, tp_info[1])

        start(timers, 'a2a-1', 0)
        y = tutel.impls.communicate.all_to_all(y, 1, 0, use_2dh=False, group=self.args.expert_parallel_group)
        stop(timers, 'a2a-1', 0)

        start(timers, 'mlp', 0)
        y = self.mlp(y)
        stop(timers, 'mlp', 0)

        start(timers, 'a2a-2', 0)
        y = tutel.impls.communicate.all_to_all(y, 0, 1, use_2dh=False, group=self.args.expert_parallel_group)
        stop(timers, 'a2a-2', 0)

        if tp_info[0] > 1:
            y = MLP_TP_G.apply(y, tp_info[1])

        y = tutel.tutel_moe.fast_decode(y.to(scores.dtype), crit, True)

        save_load_balancing_loss((tokens_per_expert, scores), self.moe_id)
        
        y = y.view(origin_shape)

        return y, self.bias

    def forward(self, x):
        # NOTE: If we're going to cast the activations to lower precision
        # do it before we permute the tokens to save bandwidth.
        x = common.cast_if_autocast_enabled(x)
        sl, bs, hs = x.size()

        # Compute the expert scores and assignments.
        scores, expert_weights, top_experts = self.router(x)

        # Compute the experts.
        x, tokens_per_expert = self.forward_fn(
            x, expert_weights, top_experts)
        save_load_balancing_loss((tokens_per_expert, scores))
        return x.view(sl, bs, hs), self.bias
