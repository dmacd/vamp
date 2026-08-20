"""Pinned public identities for the ImageNet-R-50 experiment."""

from typing import Final


IMAGENET_R_URL: Final[str] = (
    "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
)
IMAGENET_R_MD5: Final[str] = "a61312130a589d0ca1a8fca1f2bd3337"

# Frozen after downloading the public archive named above.  Preparation refuses
# to proceed until both the upstream MD5 and this stronger identity match.
IMAGENET_R_SHA256: Final[str] = (
    "18c6bf493b39a0d975d48e587437f562caab9c52ae6327dcfa9dd8eb54aa1b52"
)

TIMM_MODEL_NAME: Final[str] = "vit_base_patch16_224.augreg_in21k"
TIMM_MODEL_REPOSITORY: Final[str] = "timm/vit_base_patch16_224.augreg_in21k"
TIMM_MODEL_REVISION: Final[str] = "bb70896cd6bffdd579c3d4f7284f39d0c4ab46c3"
TIMM_MODEL_FILENAME: Final[str] = "model.safetensors"
TIMM_MODEL_SHA256: Final[str] = (
    "2d31bb5597c720912f8ea5e4f1de73909a6911798d78ae5d1b9d63a955fed049"
)

E2LORA_REPOSITORY: Final[str] = "https://github.com/kiddo127/E2-LoRA.git"
E2LORA_REVISION: Final[str] = "b4fad496d832d7a1f54a6f5dc198e71ccfbe453c"
E2LORA_CONFIG: Final[str] = "class_incremental_learning/exps/e2lora_inr_lora_50.json"

CORE_SPACE_REPOSITORY: Final[str] = (
    "https://github.com/apanariello4/core-space-merging.git"
)
CORE_SPACE_REVISION: Final[str] = "c8c0f69dd4587eaefce61414dc6ac26ee5ad31f0"

PUBLISHED_E2LORA_LAST_ACCURACY: Final[float] = 78.58
PUBLISHED_E2LORA_INCREMENTAL_ACCURACY: Final[float] = 83.96
PUBLISHED_JOINT_LAST_ACCURACY: Final[float] = 82.76

PROTOCOL_SCHEMA: Final[str] = "imagenetr50-resolved-protocol-v1"
NODE_SCHEMA: Final[str] = "imagenetr50-node-artifact-v1"
MERGE_POLICY_SCHEMA: Final[str] = "imagenetr50-merge-policy-v1"
JOB_SCHEMA: Final[str] = "imagenetr50-job-v1"
