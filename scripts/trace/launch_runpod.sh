#!/usr/bin/env bash
set -euo pipefail

: "${TRACE_NETWORK_VOLUME_ID:?Set TRACE_NETWORK_VOLUME_ID to a 150 GB high-performance volume}"

trace_image="${TRACE_RUNPOD_IMAGE:-runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04}"
trace_data_center="${TRACE_RUNPOD_DATA_CENTER:-EU-RO-1}"

runpodctl user >/dev/null
runpodctl pod create \
  --name trace-logt-vamp \
  --image "${trace_image}" \
  --gpu-id "NVIDIA GeForce RTX 4090" \
  --gpu-count 2 \
  --cloud-type SECURE \
  --data-center-ids "${trace_data_center}" \
  --container-disk-in-gb 50 \
  --network-volume-id "${TRACE_NETWORK_VOLUME_ID}" \
  --volume-mount-path /workspace \
  --ports 22/tcp \
  --wait \
  --wait-timeout 20m
