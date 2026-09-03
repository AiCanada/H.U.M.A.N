# Artificial Intelligence Canada Inc.
# Project H.U.M.A.N. V0.888
# High Utility Molecular Affinity Nexus

# Copyright 2022 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib

import torch

def is_fp16_enabled():
    # Autocast world. torch>=2.4 deprecates get_autocast_gpu_dtype() in favor
    # of get_autocast_dtype("cuda"); RBase-base was released on 2.1.2,
    # where only the old name exists. Training here is precision=32-true, so
    # this is False either way — the call just must not warn on every step.
    if hasattr(torch, "get_autocast_dtype"):
        dtype = torch.get_autocast_dtype("cuda")
    else:
        dtype = torch.get_autocast_gpu_dtype()
    try:
        enabled = torch.is_autocast_enabled("cuda")
    except TypeError:
        enabled = torch.is_autocast_enabled()
    return dtype == torch.float16 and enabled
