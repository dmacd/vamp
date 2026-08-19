"""TRACE log-t VAMP experiment contracts.

Heavy PyTorch and Hugging Face dependencies stay behind submodule boundaries so
importing :mod:`apm` retains the repository's lightweight JAX behavior.
"""

from apm.continual.trace.lineage import (
    HierarchyNode,
    HierarchyState,
    MergeEvent,
    build_hierarchy,
    empty_hierarchy,
    insert_arrival,
)
from apm.continual.trace.protocol import (
    PRIMARY_CONDITIONS,
    TASKS,
    MergePolicy,
    TraceTask,
    TrainingConfig,
    default_merge_policies,
)

__all__ = [
    "HierarchyNode",
    "HierarchyState",
    "MergeEvent",
    "MergePolicy",
    "PRIMARY_CONDITIONS",
    "TASKS",
    "TraceTask",
    "TrainingConfig",
    "build_hierarchy",
    "default_merge_policies",
    "empty_hierarchy",
    "insert_arrival",
]
