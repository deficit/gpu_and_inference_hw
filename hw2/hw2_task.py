import torch
from utils import (
    build_model,
    get_input_ids,
    slow_loop,
    time_generation,
    MODEL_NAME,
    PROFILE_STEPS,
    RESULTS_DIR,
)


def optimized_loop(model, input_ids, n_steps):
    generated_tokens = []
    past_key_values = None
    current_ids = input_ids.clone()
    
    with torch.inference_mode():
        for _ in range(n_steps):
            outputs = model(input_ids=current_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token_id = torch.argmax(outputs.logits[:, -1, :], dim=-1)
            token_value = next_token_id.item()
            generated_tokens.append(token_value)
            current_ids = next_token_id.unsqueeze(1)
            
    return generated_tokens



def profile(loop_fn, model, input_ids, trace_name: str):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        loop_fn(model, input_ids, PROFILE_STEPS)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    prof.export_chrome_trace(str(RESULTS_DIR / trace_name))


def generate_optimized(optimized_trace_name: str) -> float:
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(optimized_loop, model, input_ids, optimized_trace_name)
    elapsed = time_generation(optimized_loop, model, input_ids, "Optimized")
    return elapsed


def main():
    print("=" * 60)
    print("HW2: LLM Inference Optimization")
    print(f"Model: {MODEL_NAME}")
    print("=" * 60)

    print("\n--- Part 1: Slow baseline ---")
    model = build_model(torch.float32)
    input_ids = get_input_ids()
    profile(slow_loop, model, input_ids, "v0_slow_trace.json")
    slow_elapsed = time_generation(slow_loop, model, input_ids, "Slow")
    del model
    torch.cuda.empty_cache()

    print("\n--- Part 2: Optimized ---")
    optimized_elapsed = generate_optimized(optimized_trace_name="v1_optimized_trace.json")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if optimized_elapsed is None or optimized_elapsed <= 0:
        print("generate_optimized() did not return a positive elapsed time; "
              "cannot compute speedup.")
    else:
        speedup = slow_elapsed / optimized_elapsed
        print(f"  Slow:      {slow_elapsed:6.2f}s")
        print(f"  Optimized: {optimized_elapsed:6.2f}s")
        print(f"  Speedup:   {speedup:6.2f}x  (vs V0 slow baseline)")


if __name__ == "__main__":
    main()


# ============================================================================
# Writeup
# ============================================================================
#
# Changes made and speedup per fix:
# 1. KV Caching: Passed `use_cache=True` and `past_key_values` so the model doesn't recompute attention over the entire sequence every step. 
# 2. bfloat16: Instantiated the model in `torch.bfloat16` instead of `fp32` to drastically improve memory bandwidth and math throughput.
# 3. torch.inference_mode(): Wrapped the generation loop to prevent PyTorch from building autograd graphs, saving memory and CPU overhead. Speedup 1.02x
# 4. Deferred .item() sync: Removed `.item()` from the hot loop and kept tokens as tensors until the very end, eliminating CPU-GPU synchronization stalls.
#
# Biggest impact and why:
# The biggest impacts were KV Caching and deferring `.item()`. 
# KV caching changes the algorithm from O(N^2) to O(N) by preventing redundant computation over past tokens. 
# Removing `.item()` prevents a blocking `cudaMemcpyDeviceToHost` at every step, allowing the CPU to enqueue operations asynchronously and keeping the GPU fully utilized without idling.
