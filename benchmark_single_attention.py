import os
import argparse
from tqdm import tqdm
import torch
import triton

from models.attention.flash_attn_triton_for_hyper import flash_attn_func
from models.attention.hyper_attn import HyperAttention

try:
    from flash_attn import flash_attn_func as flash_attn_func_cuda
except ImportError:
    flash_attn_func_cuda = None


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_causal", action="store_true")
    parser.add_argument("--mode", type=str, default="fwd+bwd", choices=['fwd', 'bwd', 'fwd+bwd'])
    parser.add_argument("--attn_method", type=str, default="flash", choices=['flash', 'flash-cuda', 'hyper', 'hyper-cuda'])
    return parser.parse_args()


def get_tensors(batch_size, seq_len, head_size, dim):
    q = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn((batch_size, seq_len, head_size, dim), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    return q, k, v


def run_flash_attn(batch_size, head_size, seq_len, dim, causal, mode, impl="triton", warmup=20, rep=100):
    q, k, v = get_tensors(batch_size, seq_len, head_size, dim)
    if impl == "cuda":
        if flash_attn_func_cuda is None:
            raise ImportError("Please install flash_attn (pip install flash-attn --no-build-isolation)")
        fn = lambda: flash_attn_func_cuda(q, k, v, causal=causal)
    else:
        fn = lambda: flash_attn_func(q, k, v, None, causal, None)[0]
    if mode == 'fwd':
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, percentiles=[0.2, 0.5, 0.8])
    elif mode == 'bwd':
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, percentiles=[0.2, 0.5, 0.8])
    else: # mode == 'fwd+bwd'
        q20_fwd, median_fwd, q80_fwd = triton.testing.do_bench(fn, warmup=warmup, rep=rep, percentiles=[0.2, 0.5, 0.8])
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        q20_bwd, median_bwd, q80_bwd = triton.testing.do_bench(fn, warmup=warmup, rep=rep, percentiles=[0.2, 0.5, 0.8])
        return q20_fwd+q20_bwd, median_fwd+median_bwd, q80_fwd+q80_bwd


def run_hyper_attn(batch_size, head_size, seq_len, dim, causal, mode, impl="triton", warmup=20, rep=100):
    q, k, v = get_tensors(batch_size, head_size, seq_len, dim)
    block_size = 256
    sample_size = 256
    cuda = impl=="cuda"

    attn = HyperAttention(
        input_dim=dim,
        block_size=block_size,
        sample_size=sample_size,
        min_seq_len=4096,
        cuda=cuda).to(device='cuda', dtype=q.dtype)
    
    fn = lambda: attn(q, k, v, causal=causal)

    if mode == 'fwd':
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, percentiles=[0.2, 0.5, 0.8])
    elif mode == 'bwd':
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        return triton.testing.do_bench(fn, warmup=warmup, rep=rep, percentiles=[0.2, 0.5, 0.8])
    else: # mode == 'fwd+bwd'
        q20_fwd, median_fwd, q80_fwd = triton.testing.do_bench(fn, warmup=warmup, rep=rep, percentiles=[0.2, 0.5, 0.8])
        o = fn()
        do = torch.randn_like(o)
        fn = lambda: o.backward(do, retain_graph=True)
        q20_bwd, median_bwd, q80_bwd = triton.testing.do_bench(fn, warmup=warmup, rep=rep, percentiles=[0.2, 0.5, 0.8])
        return q20_fwd+q20_bwd, median_fwd+median_bwd, q80_fwd+q80_bwd


def main():
    args = get_arguments()
    for arg_name, arg_var in args.__dict__.items():
        print(f"{arg_name:<16} : {arg_var}")

    seq_lens = [2**i for i in range(10, 18)]
    
    attn_method = args.attn_method # ['flash', 'hyper']
    mode = args.mode # ['fwd', 'bwd', 'fwd+bwd']
    batch_size, head_size, dim = 1, 32, 64
    print(f"mode: {mode}, attn_method: {attn_method}, batch_size: {batch_size}, head_size: {head_size}, dim: {dim}")

    causal = not args.no_causal

    for seq_len in seq_lens:
        if attn_method == 'flash':
            ms = run_flash_attn(batch_size, head_size, seq_len, dim, causal, mode=args.mode)
        elif attn_method == 'flash-cuda':
            ms = run_flash_attn(batch_size, head_size, seq_len, dim, causal, mode=args.mode, impl="cuda")
        elif attn_method == 'hyper':
            ms = run_hyper_attn(batch_size, head_size, seq_len, dim, causal, mode=args.mode)
        elif attn_method == 'hyper-cuda':
            ms = run_hyper_attn(batch_size, head_size, seq_len, dim, causal, mode=args.mode, impl="cuda")
        else:
            raise NotImplementedError
        
        print(f"[{mode:<8}], {attn_method}, seq_len: {seq_len:<8}, causal: {causal}, ms: {ms[0]:.5f} ({ms[1]:.5f}, {ms[2]:.5f}) | ")


if __name__ == "__main__":
    main()


