# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""The container is the cloud bootstrap with the code baked in.

Docker is not installed on the machine these run on, so this pins what can be
checked without a daemon: the files agree with each other, nothing in the build
context drags data into the image, and the shell files are LF (both bootstrap
scripts were CRLF once, which fails on line 1 under bash).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO / "docker" / "Dockerfile"
ENTRYPOINT = REPO / "docker" / "entrypoint.sh"
COMPOSE = REPO / "docker-compose.yml"
IGNORE = REPO / ".dockerignore"

def test_dockerfile_wires_the_entrypoint_to_the_bootstrap():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY docker/entrypoint.sh /usr/local/bin/rbase-cloud" in text
    assert 'ENTRYPOINT ["rbase-cloud"]' in text
    assert "REPO=/opt/RBase" in text, "code must live outside the /workspace volume"
    assert 'VOLUME ["/workspace"]' in text
    assert "HF_REPO=AICanada/H.U.M.A.N" in text
    entry = ENTRYPOINT.read_text(encoding="utf-8")
    assert "vast_bootstrap_pdbcluster.sh" in entry
    for mode in ("smoke)", "train)", "verify)", "gpu)", "shell|bash)"):
        assert mode in entry, mode
    assert "HF_TOKEN" in entry

def test_base_image_has_blackwell_kernels():
    """The target is an RTX PRO 6000 (sm_120). torch's cu126 wheels stop at sm_90,
    so the base must be a CUDA >= 12.8 build of the validated torch line."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    match = re.search(r"ARG BASE=pytorch/pytorch:(\d+\.\d+)\.\d+-cuda(\d+)\.(\d+)-cudnn9-runtime", text)
    assert match, "base image must be an official pytorch runtime tag"
    assert match.group(1) == "2.13"
    cuda = (int(match.group(2)), int(match.group(3)))
    assert cuda >= (12, 8), f"CUDA {cuda} has no sm_120 kernels"
    entry = ENTRYPOINT.read_text(encoding="utf-8")
    assert "gpu_preflight" in entry and "get_arch_list" in entry

def test_build_context_excludes_data_runs_and_weights():
    rules = {line.strip() for line in IGNORE.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    for must in ("runs/", "runsPDB/", "rbase_cache/", "payloads/", "*.ckpt", "*.pt", "*.npy", ".git"):
        assert must in rules, must

def test_compose_gives_the_container_gpus_and_shared_memory():
    text = COMPOSE.read_text(encoding="utf-8")
    assert "ipc: host" in text, "DataLoader workers pass (L, L, 128) tensors through /dev/shm"
    assert "driver: nvidia" in text and "capabilities: [gpu]" in text
    assert "rbase_ws:/workspace" in text
    assert "HF_TOKEN: ${HF_TOKEN:?" in text

def test_docker_and_shell_files_are_lf():
    for path in (DOCKERFILE, ENTRYPOINT, COMPOSE, IGNORE,
                 REPO / "scripts" / "vast_bootstrap_pdbcluster.sh",
                 REPO / "scripts" / "vast_bootstrap.sh"):
        assert b"\r" not in path.read_bytes(), f"{path.name} has CRLF line endings"
    attrs = (REPO / ".gitattributes").read_text(encoding="utf-8")
    assert "*.sh text eol=lf" in attrs and "Dockerfile text eol=lf" in attrs
