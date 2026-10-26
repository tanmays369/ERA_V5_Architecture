"""Make a training run repeatable: the same seed must give the same loss at every step.

Everything that can introduce randomness or run-to-run differences is fixed here, once, before the model
is built:
  - the seeds of Python, NumPy and PyTorch (CPU and every GPU);
  - torch.use_deterministic_algorithms(True), so PyTorch raises an error instead of silently using an
    operation whose result can differ between runs;
  - CUBLAS_WORKSPACE_CONFIG, which cuBLAS needs to be deterministic on a GPU;
  - cuDNN autotuning off (it may pick a different kernel each run);
  - on a GPU, attention through the plain "math" kernel (the fused flash kernels are not deterministic
    in their backward pass);
  - a fixed number of CPU threads.
The data order is fixed separately, by a torch.Generator seeded from the same seed (data.py).
"""
import os
import random

import numpy as np
import torch


def make_deterministic(seed, threads=None):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if threads:
        torch.set_num_threads(int(threads))
