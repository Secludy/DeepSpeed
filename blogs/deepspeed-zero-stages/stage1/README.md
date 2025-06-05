<div align="center">

# ZeRO Stage 1: Partitioning Optimizer States

</div>

ZeRO stage&nbsp;1 reduces memory usage by partitioning the optimizer state (e.g. the Adam moments and 32-bit parameters) across all data-parallel processes. Each process keeps only a shard of these tensors, which cuts optimizer memory roughly by the data parallel degree.

The [DeepSpeed tutorial](../../docs/_tutorials/zero.md) demonstrates enabling stage&nbsp;1 with a minimal JSON configuration:

```json
{
    "zero_optimization": {
        "stage": 1,
        "reduce_bucket_size": 5e8
    }
}
```
【F:docs/_tutorials/zero.md†L48-L54】

Inside DeepSpeed, stage&nbsp;1 is implemented in `stage_1_and_2.py`. The optimizer wrapper sets `partition_gradients` to `False` to indicate ZeRO‑1 mode:

```python
# ZeRO stage 1 (False) or 2 (True)
self.partition_gradients = partition_grads
self.zero_stage_string = "ZeRO-2" if partition_grads else "ZeRO-1"
```
【F:deepspeed/runtime/zero/stage_1_and_2.py†L168-L176】

By sharding optimizer states, ZeRO stage&nbsp;1 enables training models that would otherwise exhaust GPU memory. The gradients remain replicated at this stage, so communication volume is similar to standard data parallel training.
