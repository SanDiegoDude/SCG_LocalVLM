"""
ComfyUI_QwenVL - Qwen Vision-Language Models for ComfyUI

This extension provides nodes for running Qwen2.5-VL, Qwen3-VL (vision-language),
and Qwen2.5, Qwen3 (text-only) models within ComfyUI workflows.

Supports:
- Multiple quantization levels (none, 4bit, 8bit)
- Multi-image inputs
- Video processing (via FFmpeg)
- Advanced generation parameters (temperature, top_p, top_k, min_p, repetition_penalty)
- Flash Attention 2 optimization
- ComfyUI v2 and v3 compatible

Author: Alex Cong
Repository: https://github.com/alexcong/ComfyUI_QwenVL
"""

from .nodes import QwenVL, Qwen

# Node version for ComfyUI v3 compatibility
__version__ = "2.1.3"

# ComfyUI v2/v3 compatible node registration
# These mappings work with both ComfyUI v2 and v3 architectures
NODE_CLASS_MAPPINGS = {
    "Qwen2.5VL": QwenVL,
    "Qwen2.5": Qwen,
    "QwenVL": QwenVL,      # Backwards compatibility alias
    "Qwen": Qwen,          # Backwards compatibility alias
}

# Human-readable display names for the ComfyUI interface
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2.5VL": "Qwen2.5-VL (Vision-Language)",
    "Qwen2.5": "Qwen2.5 (Text)",
    "QwenVL": "QwenVL (Vision-Language)",
    "Qwen": "Qwen (Text)",
}

# Export version and node info for ComfyUI v3 introspection
__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "__version__",
    "QwenVL",
    "Qwen",
]
