<div align="center">

# ZeRO Stage 2: Partitioning Gradients

</div>

Stage&nbsp;2 builds upon stage&nbsp;1 by also partitioning gradients after the reduce-scatter operation. Each process retains only the gradients that correspond to its optimizer shard, further lowering memory consumption and communication.

To enable stage&nbsp;2, the tutorial updates the configuration with additional optimization flags:

```json
{
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": true,
        "overlap_comm": true,
        "reduce_scatter": true,
        "reduce_bucket_size": 5e8,
        "allgather_bucket_size": 5e8
    }
}
```
【F:docs/_tutorials/zero.md†L84-L98】

DeepSpeed activates stage&nbsp;2 by setting `partition_gradients` to `True` in the optimizer wrapper. This ensures gradients are sharded in `_configure_moe_settings` and related methods:

```python
if self.partition_gradients:
    assert self.contiguous_gradients,
        "Contiguous Gradients in ZeRO Stage 2 must be set to True for MoE."
```
【F:deepspeed/runtime/zero/stage_1_and_2.py†L620-L628】

With gradients sharded, stage&nbsp;2 greatly reduces the activation memory footprint and network traffic, allowing even larger batch sizes and models.
