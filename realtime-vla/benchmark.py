import time
import argparse
import torch
from pi0_infer import Pi0Inference
from pi05_infer import Pi05Inference
from dm0_infer import DM0Inference

def benchmark_pi0(args):
    infer = Pi0Inference({
        'language_embeds' : torch.randn(args.prompt_len, 2048, dtype = torch.bfloat16),
    }, num_views=args.num_views, chunk_size=args.chunk_size)

    input_image = torch.randn(args.num_views, 224, 224, 3, dtype = torch.bfloat16).cuda()
    input_state = torch.randn(32, dtype = torch.bfloat16).cuda()
    input_noise = torch.randn(args.chunk_size, 32, dtype = torch.bfloat16).cuda()

    # Warm up
    for _ in range(3):
        _ = infer.forward(input_image, input_state, input_noise)
        torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    times = []
    for _ in range(iterations):
        t0 = time.time()
        _ = infer.forward(input_image, input_state, input_noise)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

    print('[Pi0 Triton]:', 'views', args.num_views, 'prompt_len', args.prompt_len, 'chunk_size', args.chunk_size)
    print('runs', len(times), 'median time per inference:', '%.3f'%(sorted(times)[len(times)//2]*1000), 'ms')

def benchmark_pi05(args):
    infer = Pi05Inference({
        'language_embeds' : torch.randn(args.prompt_len, 2048, dtype = torch.bfloat16),
    }, num_views=args.num_views, chunk_size=args.chunk_size, discrete_state_input=False)

    input_image = torch.randn(args.num_views, 224, 224, 3, dtype=torch.bfloat16, device="cuda")
    input_noise = torch.randn(args.chunk_size, 32, dtype=torch.bfloat16, device="cuda")

    # Warm up
    for _ in range(3):
        _ = infer.forward(input_image, input_noise)
        torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    times = []
    for _ in range(iterations):
        t0 = time.time()
        _ = infer.forward(input_image, input_noise)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

    print('[Pi05 Triton]:', 'views', args.num_views, 'chunk_size', args.chunk_size)
    print('runs', len(times), 'median time per inference:', '%.3f'%(sorted(times)[len(times)//2]*1000), 'ms')

def benchmark_dm0(args):
    infer = DM0Inference({}, num_images=args.num_views, max_lang_len=100)
    input_image = torch.randn(args.num_views, 3, 728, 728, dtype = torch.bfloat16).cuda()
    input_ids = torch.randint(0, 1000, (100,), dtype = torch.bfloat16).cuda()
    input_noise = torch.randn(args.chunk_size, 32, dtype = torch.bfloat16).cuda()

    # Warm up
    for _ in range(3):
        _ = infer.forward(input_image, input_ids, input_noise)
        torch.cuda.synchronize()

    # Benchmark
    iterations = 100
    times = []
    for _ in range(iterations):
        t0 = time.time()
        _ = infer.forward(input_image, input_ids, input_noise)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

    print('[DM0 Triton]:', 'num_images', args.num_views, 'max_lang_len', 100, 'chunk_size', args.chunk_size)
    print('runs', len(times), 'median time per inference:', '%.3f'%(sorted(times)[len(times)//2]*1000), 'ms')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_version", type=str, choices=["pi0", "pi05", "dm0"], default="dm0")

    parser.add_argument("--num_views", type=int, default=3, help="Number of views")
    parser.add_argument("--chunk_size", type=int, default=50, help="Chunk size")
    parser.add_argument("--prompt_len", type=int, default=0, help="Pi0 prompt length")

    args = parser.parse_args()

    if args.model_version == "pi0":
        benchmark_pi0(args)
    elif args.model_version == "pi05":
        benchmark_pi05(args)
    elif args.model_version == "dm0":
        benchmark_dm0(args)

if __name__ == "__main__":
    main()