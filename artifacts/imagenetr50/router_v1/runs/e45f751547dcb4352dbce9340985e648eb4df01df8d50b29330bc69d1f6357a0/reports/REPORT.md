# ImageNet-R-50 Recursive Learned Router Oracle Recovery

Router protocol: `e45f751547dcb4352dbce9340985e648eb4df01df8d50b29330bc69d1f6357a0`

The sealed inference run is a read-only dependency. Learned-router jobs report zero leaf and zero inference-parent optimizer steps; R3 is a predeclared main architecture, paired with R1 rather than activated after seeing test results.

## Validation capacity gate

The preregistered validation capacity gate closed. Neither R1 nor R3 finished within 1.0 percentage point of the I-U100 true-node oracle. A4 ran as the declared nonlinear diagnostic; the recursive B/C matrix, test split, and additional seeds were not run.

| condition_id | architecture | maintenance | routed_accuracy | oracle_accuracy | oracle_gap | selection_accuracy | top2_selection_accuracy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A0 | centroid | existing | 64.562 | 97.646 | 33.083 | 65.458 | 81.417 |
| A1 | r0 | flat_full | 59.083 | 97.646 | 38.563 | 60.167 | 80.833 |
| A2 | r1 | flat_full | 58.729 | 97.646 | 38.917 | 59.750 | 80.229 |
| A3 | r3 | flat_full | 57.750 | 97.646 | 39.896 | 58.854 | 80.187 |
| A4 | r2 | flat_full | 59.458 | 97.646 | 38.187 | 60.604 | 81.563 |

## Final test results

_No completed rows._

## Paired R3 minus R1

| inference_condition | maintenance | router_seed | split | r3_minus_r1_accuracy | paired_ci95_lower | paired_ci95_upper | r1_only_correct | r3_only_correct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I-U100 | flat_full | 1993 | validation | -0.979 | -1.812 | -0.167 | 224 | 177 |

## Resource accounting

| condition | architecture | maintenance | router_seed | final_live_nodes | learned_router_parameters | response_kernel_bytes | router_optimizer_steps | leaf_optimizer_steps | inference_parent_optimizer_steps | candidate_adapted_forwards |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I-U100 | r1 | svd0 | 1993 | 4 | 32260 | 0 | 386 | 0 | 0 | 0 |
| I-U100 | r1 | exact | 1993 | 4 | 16130 | 0 | 264 | 0 | 0 | 0 |
| I-U100 | r3 | svd5 | 1993 | 4 | 32292 | 1575936 | 425 | 0 | 0 | 0 |
| I-U100 | r3 | exact | 1993 | 4 | 16146 | 787968 | 276 | 0 | 0 | 0 |
| I-U100 | r3 | u100 | 1993 | 4 | 32292 | 1575936 | 1897 | 0 | 0 | 0 |
| I-U100 | r3 | svd0 | 1993 | 4 | 32292 | 1575936 | 413 | 0 | 0 | 0 |
| I-U100 | r1 | svd5 | 1993 | 4 | 32260 | 0 | 564 | 0 | 0 | 0 |
| I-U100 | r1 | u100 | 1993 | 4 | 32260 | 0 | 2039 | 0 | 0 | 0 |

## Durable status

```json
{
  "scheduler": {
    "counts": {
      "COMPLETE": 13,
      "FAILED": 0,
      "PAUSED": 0,
      "PENDING": 0,
      "RUNNING": 0
    },
    "jobs": 13,
    "run_hash": "e45f751547dcb4352dbce9340985e648eb4df01df8d50b29330bc69d1f6357a0",
    "running": [],
    "schema_version": "imagenetr50-scheduler-summary-v1"
  },
  "workflow": {
    "gate_open": false,
    "phase": "COMPLETE_CAPACITY_FAILURE",
    "router_run_hash": "e45f751547dcb4352dbce9340985e648eb4df01df8d50b29330bc69d1f6357a0",
    "schema_version": "imagenetr50-router-workflow-state-v1"
  }
}
```
