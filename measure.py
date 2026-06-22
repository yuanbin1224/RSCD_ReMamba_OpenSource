import argparse
import contextlib
import csv
import os
import time

import torch
import torch.nn as nn

from model import ReMambaNet


class LogitWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x1, x2):
        output = self.model(x1, x2, return_features=False)
        if isinstance(output, dict):
            return output["change_pred"]
        if isinstance(output, (tuple, list)):
            return output[0]
        return output


def count_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def params_to_mb(num_params, dtype_size=4):
    return num_params * dtype_size / 1024 / 1024


def state_dict_size_mb(model):
    total_bytes = 0
    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()
    return total_bytes / 1024 / 1024


def module_param_breakdown(model):
    total_params, _ = count_params(model)

    rows = []

    def append(name, module):
        params = sum(p.numel() for p in module.parameters())
        ratio = params / total_params * 100.0 if total_params else 0.0
        rows.append((name, params, ratio, params_to_mb(params)))

    append("encoder.shared_resnet", model.encoder)
    append("feature_preparation", model.prepare)
    for idx, module in enumerate(model.prepare, 1):
        append(f"prepare_l{idx}", module)

    append("recgi_total", model.interaction)
    for idx, module in enumerate(model.interaction, 1):
        append(f"recgi_l{idx}", module)

    append("decoder", model.decoder)
    return rows


def load_checkpoint(model, checkpoint_path, device):
    if not checkpoint_path:
        return
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    cleaned = {}
    for key, value in checkpoint.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[Load Warning] Missing keys: {len(missing)}")
    if unexpected:
        print(f"[Load Warning] Unexpected keys: {len(unexpected)}")


def measure_flops(model, device, image_size=512, batch_size=1):
    x1 = torch.randn(batch_size, 3, image_size, image_size, device=device)
    x2 = torch.randn(batch_size, 3, image_size, image_size, device=device)
    wrapped = LogitWrapper(model).eval()

    try:
        from fvcore.nn import FlopCountAnalysis

        with torch.no_grad():
            analysis = FlopCountAnalysis(wrapped, (x1, x2))
            flops = analysis.total()
            unsupported = analysis.unsupported_ops()
        return flops, "fvcore", dict(unsupported)
    except Exception as fvcore_error:
        try:
            from thop import profile

            with torch.no_grad():
                flops, _ = profile(wrapped, inputs=(x1, x2), verbose=False)
            return flops, "thop", {}
        except Exception as thop_error:
            print(f"[Warning] FLOPs measurement failed. fvcore: {fvcore_error}; thop: {thop_error}")
            return 0, "none", {}


def amp_context(device, enabled=True):
    if enabled and device.type == "cuda":
        try:
            return torch.amp.autocast("cuda", dtype=torch.float16)
        except AttributeError:
            return torch.cuda.amp.autocast(dtype=torch.float16)
    return contextlib.nullcontext()


def measure_latency_and_fps(
    model,
    device,
    image_size=512,
    batch_size=1,
    warmup=50,
    iterations=100,
    use_amp=True,
    use_compile=False,
    channels_last=True,
):
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    wrapped = LogitWrapper(model).eval()
    if channels_last and device.type == "cuda":
        wrapped = wrapped.to(memory_format=torch.channels_last)

    if use_compile and hasattr(torch, "compile"):
        wrapped = torch.compile(wrapped)

    memory_format = torch.channels_last if channels_last and device.type == "cuda" else torch.contiguous_format
    x1 = torch.randn(batch_size, 3, image_size, image_size, device=device).contiguous(memory_format=memory_format)
    x2 = torch.randn(batch_size, 3, image_size, image_size, device=device).contiguous(memory_format=memory_format)

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        with amp_context(device, enabled=use_amp):
            for _ in range(warmup):
                _ = wrapped(x1, x2)

        if device.type == "cuda":
            torch.cuda.synchronize()

        if device.type == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            with amp_context(device, enabled=use_amp):
                for _ in range(iterations):
                    _ = wrapped(x1, x2)
            ender.record()
            torch.cuda.synchronize()
            total_ms = starter.elapsed_time(ender)
        else:
            start_time = time.perf_counter()
            with amp_context(device, enabled=use_amp):
                for _ in range(iterations):
                    _ = wrapped(x1, x2)
            total_ms = (time.perf_counter() - start_time) * 1000.0

    avg_latency_ms = total_ms / max(iterations, 1)
    p50_latency_ms = avg_latency_ms
    p95_latency_ms = avg_latency_ms
    fps = batch_size * 1000.0 / max(avg_latency_ms, 1e-9)
    peak_memory_mb = 0.0
    if device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "avg_latency_ms": avg_latency_ms,
        "p50_latency_ms": p50_latency_ms,
        "p95_latency_ms": p95_latency_ms,
        "fps": fps,
        "peak_memory_mb": peak_memory_mb,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Measure ReMamba model scale and inference efficiency.")
    parser.add_argument("--gpu_id", default="0", type=str)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--checkpoint", "--model_path", dest="checkpoint", default="", type=str)
    parser.add_argument("--save_csv", default="", type=str)

    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--scan_backend", choices=["fast", "torch"], default="fast")
    parser.add_argument("--ssm_d_state", type=int, default=8)
    parser.add_argument("--ssm_ratio", type=float, default=1.0)
    parser.add_argument("--lambda_reweight", type=float, default=1.0)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--channels_last", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_flops", action="store_true")
    parser.add_argument("--skip_speed", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ReMambaNet(
        pretrained=args.pretrained,
        scan_backend=args.scan_backend,
        ssm_d_state=args.ssm_d_state,
        ssm_ratio=args.ssm_ratio,
        lambda_reweight=args.lambda_reweight,
    ).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    total_params, trainable_params = count_params(model)
    non_trainable_params = total_params - trainable_params
    model_size_mb = params_to_mb(total_params)
    state_size_mb = state_dict_size_mb(model)

    flops = 0
    flops_backend = "skipped"
    unsupported_ops = {}
    if not args.skip_flops:
        flops, flops_backend, unsupported_ops = measure_flops(
            model,
            device=device,
            image_size=args.image_size,
            batch_size=args.batch_size,
        )

    speed = {
        "avg_latency_ms": 0.0,
        "p50_latency_ms": 0.0,
        "p95_latency_ms": 0.0,
        "fps": 0.0,
        "peak_memory_mb": 0.0,
    }
    if not args.skip_speed:
        speed = measure_latency_and_fps(
            model,
            device=device,
            image_size=args.image_size,
            batch_size=args.batch_size,
            warmup=args.warmup,
            iterations=args.iterations,
            use_amp=args.amp,
            use_compile=args.compile,
            channels_last=args.channels_last,
        )

    print("=" * 60)
    print("ReMamba Complexity and Efficiency Report")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Input: batch={args.batch_size}, shape=3x{args.image_size}x{args.image_size} for each temporal image")
    print(f"Total Params: {total_params:,} ({model_size_mb:.2f} MB, fp32)")
    print(f"Trainable Params: {trainable_params:,}")
    print(f"Non-trainable Params: {non_trainable_params:,}")
    print(f"State Dict Size: {state_size_mb:.2f} MB")
    print("-" * 60)
    print("Parameter Breakdown:")
    for name, params, ratio, size_mb in module_param_breakdown(model):
        print(f"  {name:<24} {params:>12,}  {ratio:>6.2f}%  {size_mb:>8.2f} MB")
    print("-" * 60)
    if flops > 0:
        print(f"FLOPs Backend: {flops_backend}")
        print(f"FLOPs: {flops:,} ({flops / 1e9:.4f} GFLOPs)")
    else:
        print(f"FLOPs Backend: {flops_backend}")
        print("FLOPs: N/A")
    if unsupported_ops:
        print(f"Unsupported Ops: {unsupported_ops}")
    if not args.skip_speed:
        print(f"Timing Mode: aggregate after warmup ({args.warmup} warmup, {args.iterations} timed)")
        print(f"Avg Latency: {speed['avg_latency_ms']:.3f} ms")
        print(f"P50 Latency: {speed['p50_latency_ms']:.3f} ms (same as aggregate avg)")
        print(f"P95 Latency: {speed['p95_latency_ms']:.3f} ms (same as aggregate avg)")
        print(f"FPS: {speed['fps']:.2f}")
        if device.type == "cuda":
            print(f"Peak CUDA Memory: {speed['peak_memory_mb']:.2f} MB")
    print("=" * 60)

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        row = {
            "model": "ReMambaNet",
            "device": str(device),
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "non_trainable_params": non_trainable_params,
            "model_size_mb_fp32": model_size_mb,
            "state_dict_size_mb": state_size_mb,
            "flops": flops,
            "gflops": flops / 1e9 if flops else 0.0,
            "flops_backend": flops_backend,
            "avg_latency_ms": speed["avg_latency_ms"],
            "p50_latency_ms": speed["p50_latency_ms"],
            "p95_latency_ms": speed["p95_latency_ms"],
            "fps": speed["fps"],
            "peak_memory_mb": speed["peak_memory_mb"],
            "amp": args.amp,
            "compile": args.compile,
            "channels_last": args.channels_last,
            "checkpoint": args.checkpoint,
        }
        write_header = not os.path.exists(args.save_csv)
        with open(args.save_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        print(f"CSV saved to: {args.save_csv}")


if __name__ == "__main__":
    main()
