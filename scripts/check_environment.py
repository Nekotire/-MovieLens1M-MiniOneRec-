#!/usr/bin/env python3
import platform
import shutil
import sys

import torch


errors = []
if platform.system() != "Linux":
    errors.append(f"expected Linux, found {platform.system()}")
if sys.version_info[:2] != (3, 11):
    errors.append(f"expected Python 3.11, found {platform.python_version()}")
if shutil.which("nvidia-smi") is None:
    errors.append("nvidia-smi is not available")
if not torch.__version__.startswith("2.6.0"):
    errors.append(f"expected PyTorch 2.6.0, found {torch.__version__}")
if not torch.cuda.is_available():
    errors.append("PyTorch CUDA is unavailable")
if errors:
    raise SystemExit("Environment check failed:\n- " + "\n- ".join(errors))
print(f"Environment PASS: Linux, Python {platform.python_version()}, PyTorch {torch.__version__}, CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
