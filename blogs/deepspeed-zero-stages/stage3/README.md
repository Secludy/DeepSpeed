<div align="center">

# ZeRO Stage 3: Partitioning Parameters and Offloading

</div>

Stage&nbsp;3 partitions the remaining model parameters themselves across data-parallel processes. Only the needed shards are gathered during the forward and backward passes, leading to linear memory scaling with the number of GPUs. Stage&nbsp;3 also unlocks offloading to CPU or NVMe, known as ZeRO‑Infinity.

A typical configuration enabling ZeRO‑3 with offloading looks like this:

```json
{
    "zero_optimization": {
        "stage": 3,
        "contiguous_gradients": true,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_prefetch_bucket_size": 1e7,
        "stage3_param_persistence_threshold": 1e5,
        "reduce_bucket_size": 1e7,
        "sub_group_size": 1e9,
        "offload_optimizer": {
            "device": "cpu"
         },
        "offload_param": {
            "device": "cpu"
       }
   }
}
```
【F:docs/_tutorials/zero.md†L129-L147】

Within the stage&nbsp;3 optimizer, parameters are partitioned and managed by groups of processes:

```python
#num of ranks in a ZeRO param partitioning group
self.zero_hpz_partition_size = zero_hpz_partition_size
print_rank_0(
    f"ZeRO Stage 3 param partitioning group {self.zero_hpz_partition_size} {zero_param_parallel_group}",
    force=False)
```
【F:deepspeed/runtime/zero/stage3.py†L208-L216】

Utilities such as `estimate_zero3_model_states_mem_needs_all_live` help estimate memory requirements for a given model and hardware setup:

```python
def estimate_zero3_model_states_mem_needs_all_live(model,
                                                   num_gpus_per_node=1,
                                                   num_nodes=1,
                                                   additional_buffer_factor=1.5):
    """Print out estimates on memory usage requirements for ZeRO 3 params, optim states and gradients"""
```
【F:deepspeed/runtime/zero/stage3.py†L3148-L3159】

By partitioning all training states and optionally offloading them, ZeRO stage&nbsp;3 enables training trillion-parameter models on commodity hardware.
