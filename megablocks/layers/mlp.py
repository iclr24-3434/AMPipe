from megablocks.layers import common
from megablocks.layers import mpu
from megablocks.layers.arguments import Arguments, InitFn
from megablocks.layers.gelu import gelu
import stk
import torch
import torch.nn.functional as F


def create_moe_expert_weights(args : Arguments,
                              num_experts : int,
                              ffn_hidden_size : int,
                              hidden_size : int,
                              init_method : InitFn):
    # Create the entire weight matrix such that the sampled weights will
    # not vary between data parallelism and expert model parallelism for
    # the same random seed.
    master_weights = torch.empty(
        num_experts, ffn_hidden_size, hidden_size,
        device=args.device,
        dtype=common.dtype(args))
    init_method(master_weights)

    if not args.moe_expert_model_parallelism:
        return master_weights

    # Calculate the amount of sharding in each dimension.
    expert_sharding_degree = mpu.expert_sharding_degree(args)
    hidden_sharding_degree = mpu.hidden_sharding_degree(args)

    # Calculate the experts per rank.
    #
    # NOTE: We assign ranks to be expert parallel before going
    # tensor parallel.
    rank = mpu.get_expert_parallel_rank(args)
    expert_rank = rank % expert_sharding_degree
    num_experts_per_rank = num_experts // expert_sharding_degree
    start_expert = expert_rank * num_experts_per_rank
    end_expert = (expert_rank + 1) * num_experts_per_rank

    # Calculate the rows per rank.
    row_rank = rank // expert_sharding_degree
    num_rows_per_rank = ffn_hidden_size // hidden_sharding_degree
    start_row = row_rank * num_rows_per_rank
    end_row = (row_rank + 1) * num_rows_per_rank

    # Slice the weight matrix to get the chunk for this rank.
    with torch.no_grad():
        weights = master_weights[
            start_expert:end_expert, start_row:end_row]
    return weights


class MLP(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        expert_parallel_world_size = mpu.get_expert_parallel_world_size(args)
        experts_per_rank = mpu.experts_per_rank(args)

        
        self.w1 = torch.nn.Parameter(torch.empty(
            experts_per_rank,
            args.hidden_size,
            mpu.features_per_rank(args) // args.tensor_model_parallel_size,
            device=args.device,
            dtype=common.dtype(args)))
        self.w2 = torch.nn.Parameter(torch.empty(
            experts_per_rank,
            mpu.features_per_rank(args) // args.tensor_model_parallel_size,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        mpu.set_expert_model_parallel_attributes(
            self.w1, args.moe_expert_model_parallelism)
        mpu.set_expert_model_parallel_attributes(
            self.w2, args.moe_expert_model_parallelism)

        
        # Initialize the parameters for the MLP.
        #
        # NOTE: It is important that we create the weight tensors prior
        # to creating the master weights and slicing our the piece for
        # this rank. If the master weights are created first the PyTorch
        # caching allocator appears to use the same memory block for these
        # and the slice which causes large increases in our peak memory
        # usage.
        if args.tensor_model_parallel_size == 1: 
            with torch.no_grad():
                w1 = create_moe_expert_weights(
                    args, args.moe_num_experts, args.ffn_hidden_size  // args.tensor_model_parallel_size,
                    args.hidden_size, args.init_method)
                self.w1.copy_(w1.transpose(1, 2).contiguous())
                self.w2.copy_(create_moe_expert_weights(
                    args, args.moe_num_experts, args.ffn_hidden_size // args.tensor_model_parallel_size,
                    args.hidden_size, args.output_layer_init_method))
        else:
            with torch.no_grad():
                tp_rank = torch.distributed.get_rank(args.tensor_model_parallel_group)
                w1 = create_moe_expert_weights(
                    args, args.moe_num_experts, args.ffn_hidden_size, #// args.tensor_model_parallel_size,
                    args.hidden_size, args.init_method)
                sizes = args.ffn_hidden_size // args.tensor_model_parallel_size

                self.w1.copy_(w1.transpose(1, 2)[:,:,tp_rank * sizes:(tp_rank + 1) * sizes].contiguous())
                w2 = create_moe_expert_weights(
                    args, args.moe_num_experts, args.ffn_hidden_size, #// args.tensor_model_parallel_size,
                    args.hidden_size, args.output_layer_init_method)

                self.w2.copy_(w2[:,tp_rank * sizes:(tp_rank + 1) * sizes,:])
            

    def forward(self, x):
        return torch.bmm(F.gelu(
            torch.bmm(x, self.w1), approximate="tanh"), self.w2)


def _gather_weights(w, group, async_op=False):
    n, k = w.shape
    world_size = torch.distributed.get_world_size(group)
    parallel_w = torch.empty(
        n * world_size, k, device=w.device, dtype=w.dtype)
    handle = torch.distributed.all_gather_into_tensor(
        parallel_w, w, group=group, async_op=async_op)
    return parallel_w, handle


def _scaled_reduce_scatter(parallel_dw, group, async_op=False):
    n, k = parallel_dw.shape
    world_size = torch.distributed.get_world_size(group)
    assert (n % world_size) == 0

    # Pre-scale the gradients by the world size.
    #
    # NOTE: Reduce in float32, always.
    parallel_dw = parallel_dw.float() / world_size

    dw = torch.empty(
        n // world_size, k,
        device=parallel_dw.device,
        dtype=torch.float32)
    handle = torch.distributed.reduce_scatter_tensor(
        dw, parallel_dw, group=group, async_op=async_op)
    return dw, handle


class WeightParallelSddNt(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, x, w, topo, group):
        # [m, k] x [n, k] = [m, n]
        if not x.is_contiguous() or not w.is_contiguous():
            raise ValueError("Expected contiguous 'x' and 'w'.")

        ctx.group = group
        ctx.shape = topo.shape
        ctx.save_for_backward(
            x, w,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t)

        # TODO(tgale): Support prefetching forward weights.
        parallel_w, _ = _gather_weights(w, group)
        return stk.ops.sdd(x, parallel_w.t(), topo).data

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        x, w = ctx.saved_tensors[:2]
        grad = stk.Matrix(ctx.shape, grad, *ctx.saved_tensors[2:])

        # Start the weight gather asynchronously to overlap with the
        # weight gradient computation.
        parallel_w, handle = _gather_weights(w, ctx.group, async_op=True)
        parallel_dw = None
        if ctx.needs_input_grad[1]:
            parallel_dw = stk.ops.dsd(grad.t(), x)

        # Start the weight gradient reduce scatter to overlap with the
        # data gradient computation.
        handle.wait()
        dw, handle = _scaled_reduce_scatter(parallel_dw, ctx.group, async_op=True)
        dx = None
        if ctx.needs_input_grad[0]:
            dx = stk.ops.dsd(grad, parallel_w)

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        handle.wait()
        dw = dw.to(w.dtype)
        return dx, dw, None, None


def weight_parallel_sdd_nt(a, b, topo, group):
    return stk.Matrix(
        topo.size(),
        WeightParallelSddNt.apply(a, b, topo, group),
        topo.row_indices,
        topo.column_indices,
        topo.offsets,
        topo.column_indices_t,
        topo.offsets_t,
        topo.block_offsets_t)


class WeightParallelDsdNn(torch.autograd.Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx,
                shape,
                data,
                row_indices,
                column_indices,
                offsets,
                column_indices_t,
                offsets_t,
                block_offsets_t,
                w,
                group):
        # [m, k] x [k, n] = [m, n]
        if not data.is_contiguous() or not w.is_contiguous():
            raise ValueError("Expected contiguous 'data' and 'w'.")

        ctx.group = group
        ctx.shape = shape
        ctx.save_for_backward(
            data,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t,
            w)
        x = stk.Matrix(
            shape,
            data,
            row_indices,
            column_indices,
            offsets,
            column_indices_t,
            offsets_t,
            block_offsets_t)

        # TODO(tgale): Support prefetching forward weights.
        parallel_w, _ = _gather_weights(w, group)
        return stk.ops.dsd(x, parallel_w)

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad):
        x = stk.Matrix(ctx.shape, *ctx.saved_tensors[:-1])
        w = ctx.saved_tensors[-1]

        # Start the weight gather asynchronously to overlap with the
        # weight gradient computation.
        parallel_w, handle = _gather_weights(w, ctx.group, async_op=True)
        parallel_dw = None
        if ctx.needs_input_grad[-2]:
            parallel_dw = stk.ops.dsd(x.t(), grad)

        # Start the weight gradient reduce scatter to overlap with the
        # data gradient computation.
        handle.wait()
        dw, handle = _scaled_reduce_scatter(parallel_dw, ctx.group, async_op=True)
        dx = None
        if ctx.needs_input_grad[1]:
            dx = stk.ops.sdd(grad, parallel_w.t(), x)

        # NOTE: Be careful to wait and only cast dw to the output dtype once
        # we've blocked on the asynchronous NCCL operation.
        handle.wait()
        dw = dw.to(w.dtype)
        return None, dx.data, None, None, None, None, None, None, dw, None


def weight_parallel_dsd_nn(a, b, group):
    return WeightParallelDsdNn.apply(
        a.size(),
        a.data,
        a.row_indices,
        a.column_indices,
        a.offsets,
        a.column_indices_t,
        a.offsets_t,
        a.block_offsets_t,
        b,
        group)


def create_dmoe_expert_weights(args : Arguments,
                               num_experts : int,
                               rows : int,
                               columns : int,
                               init_method : InitFn):
    weights = create_moe_expert_weights(
        args, num_experts, rows, columns, init_method)
    weights = weights.view([-1, columns])
    rows, columns = weights.shape

    if not args.moe_weight_parallelism:
        return weights

    # Caclculate the number of rows on this weight parallel partition.
    # 'rows' must be divisible by weight parallel world size.
    weight_parallel_world_size = mpu.get_weight_parallel_world_size(args)
    assert (rows % weight_parallel_world_size) == 0
    num_rows_per_rank = rows // weight_parallel_world_size
    rank = mpu.get_weight_parallel_rank(args)
    start_row = rank * num_rows_per_rank
    end_row = (rank + 1) * num_rows_per_rank
    return weights[start_row:end_row]


class SparseMLP(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args
        num_rows_per_rank = (
            (mpu.experts_per_rank(args) * mpu.features_per_rank(args)) //
            mpu.get_weight_parallel_world_size(args)
        )

        self.w1 = torch.nn.Parameter(torch.empty(
            num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        self.w2 = torch.nn.Parameter(torch.empty(
            num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))

        # Initialize the parameters for the MLP.
        #
        # NOTE: It is important that we create the weight tensors prior
        # to creating the master weights and slicing our the piece for
        # this rank. If the master weights are created first the PyTorch
        # caching allocator appears to use the same memory block for these
        # and the slice which causes large increases in our peak memory
        # usage.
        with torch.no_grad():
            self.w1.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.init_method))
            self.w2.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.output_layer_init_method))

        should_set_attribute = (
            args.moe_expert_model_parallelism or args.moe_weight_parallelism)
        mpu.set_expert_model_parallel_attributes(
            self.w1, should_set_attribute)
        mpu.set_expert_model_parallel_attributes(
            self.w2, should_set_attribute)

    def parallel_forward(self, x, topo):
        x = weight_parallel_sdd_nt(
            x, self.w1, topo, self.args.weight_parallel_group)
        return weight_parallel_dsd_nn(
            gelu(x), self.w2, self.args.weight_parallel_group)

    def forward(self, x, topo):
        if self.args.moe_weight_parallelism:
            return self.parallel_forward(x, topo)
        x = stk.ops.sdd(x, self.w1.t(), topo)
        return stk.ops.dsd(gelu(x), self.w2)
