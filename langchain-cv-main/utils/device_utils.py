# utils/device_utils.py
import torch

def get_device():
    """
    Returns the most appropriate device for PyTorch computations.
    """
    if torch.cuda.is_available():
        return "cuda"  # NVIDIA GPU
    elif torch.backends.mps.is_available():
        return "mps"  # macOS Metal Performance Shaders
    else:
        return "cpu"  # Fallback to CPU
