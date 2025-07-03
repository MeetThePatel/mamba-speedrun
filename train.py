import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from mamba_ssm import Mamba2
from torch.nn.parallel import DistributedDataParallel as DDP

from data.dataloader import PAD_TOKEN, MambaDataloader


@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2  # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A  # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[torch.Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size], update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: torch.Tensor = group["update_buffer"]
            update_buffer_views: list[torch.Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[torch.Tensor] = group["params"]
            handle = None
            params_world = None

            def update_prev():  # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.add_(g_world.view_as(p_world), alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5)

            for base_i in range(len(params))[:: self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: torch.Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev()  # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()


class MambaLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, d_state: int, d_conv: int, expand: int, n_layers: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.n_layers = n_layers

        assert d_model % expand == 0, "d_model must be divisible by expand."
        headdim = d_model // expand
        assert d_state % headdim == 0, "d_state must be divisible by d_model//expand."

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_length]
        x = self.embedding(x)
        # x: [batch_size, seq_length, d_model]
        for layer in self.layers:
            x = layer(x)
        # x: [batch_size, seq_length, d_model]
        out = self.out_projection(x)
        # out: [batch_size, seq_length, vocab_size]
        return out


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", rank)

    model = MambaLanguageModel(vocab_size=50277, d_model=512, d_state=768, d_conv=4, expand=4, n_layers=16)
    model.to(device)
    model = DDP(model, device_ids=[rank])

    print(f"{sum(p.numel() for p in model.module.parameters()):,}")

    hidden_matrix_params = [p for n, p in model.module.named_parameters() if p.ndim >= 2 and "embed" not in n]

    embed_params = [p for n, p in model.module.named_parameters() if "embed" in n]
    scalar_params = [p for p in model.module.parameters() if p.ndim < 2]
    head_params = [model.module.out_projection.weight]
    adam_params = [dict(params=head_params, lr=0.003), dict(params=embed_params, lr=0.003), dict(params=scalar_params, lr=0.003)]

    optimizer1 = torch.optim.Adam(adam_params, eps=1e-10, amsgrad=True, fused=True)
    optimizer2 = Muon(hidden_matrix_params, lr=0.004, momentum=0.95, rank=rank, world_size=world_size)
    loss_function = torch.nn.CrossEntropyLoss(ignore_index=PAD_TOKEN, reduction="sum")

    dataloader = MambaDataloader(data_directory="./data/fineweb10B_bucketed", rank=rank, seed=42)

    n_steps = 10000
    step = 0
    log_interval = 1

    total_train_start_time = time.perf_counter()
    log_start_time = time.perf_counter()

    model.train()

    while step <= n_steps:
        step_start_time = time.perf_counter()

        batch = dataloader.next_batch(batch_size=16, sequence_length=512)
        batch = batch.pin_memory().to(device, non_blocking=True)

        inputs = batch[:, :-1]
        targets = batch[:, 1:]

        outputs: torch.Tensor = model(inputs)

        lhs = outputs.reshape(-1, outputs.size(-1))  # Reshape to [N, C] where N=B*L, C=vocab_size
        rhs = targets.reshape(-1)  # Reshape to [N]

        loss: torch.Tensor = loss_function(lhs, rhs)
        loss.backward()

        optimizer1.step()
        optimizer2.step()

        optimizer1.zero_grad(set_to_none=True)
        optimizer2.zero_grad(set_to_none=True)

        step += 1

        if step % log_interval == 0:
            dist.barrier()

            if rank == 0:
                step_elapsed = time.perf_counter() - step_start_time
                total_elapsed = time.perf_counter() - total_train_start_time

                print(
                    f"Step: {step:5d}/{n_steps} | "
                    f"Loss: {loss.item():.3f} | "
                    f"Step Time: {step_elapsed * 1000:.3f} ms | "
                    f"Total Train Time: {total_elapsed:.3f} s"
                )

    dist.destroy_process_group()
