"""
Attention Autograd Wrapper Builder

Factory utilities constructing high-performance autograd wrappers and analytical backward passes for attention kernels.

Author: Taha Majlesi
"""

import torch

def make_attention_autograd_fn(forward_dispatch_fn, ref_fn, grad_arg_names, has_kwargs=False):
    class _AttentionAutogradFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, *args):
            ctx.arg_names = grad_arg_names
            ctx.saved_arg_indices = []
            saved = []
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    saved.append(arg)
                    ctx.saved_arg_indices.append(i)
                else:
                    saved.append(None)
            ctx.save_for_backward(*saved)
            ctx.num_total_args = len(args)
            return forward_dispatch_fn(*args)

        @staticmethod
        def backward(ctx, grad_output):
            saved = list(ctx.saved_tensors)
            args_for_ref = []
            tensors_to_diff = []
            tensor_indices = []

            for i in range(ctx.num_total_args):
                arg = saved[i]
                if arg is not None and isinstance(arg, torch.Tensor):
                    if arg.requires_grad:
                        t = arg.detach().requires_grad_(True)
                        args_for_ref.append(t)
                        tensors_to_diff.append(t)
                        tensor_indices.append(i)
                    else:
                        args_for_ref.append(arg.detach())
                else:
                    args_for_ref.append(arg)

            with torch.enable_grad():
                ref_out = ref_fn(*args_for_ref)
                grads = torch.autograd.grad(ref_out, tensors_to_diff, grad_output)

            result_grads = [None] * ctx.num_total_args
            for idx, g in zip(tensor_indices, grads):
                result_grads[idx] = g

            return tuple(result_grads)

    return _AttentionAutogradFunction

