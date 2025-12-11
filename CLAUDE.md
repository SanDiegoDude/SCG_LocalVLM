# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI_QwenVL is a custom node extension for ComfyUI that implements Qwen2.5-VL and Qwen3-VL vision-language models alongside Qwen2.5 text-only models. It provides multimodal AI capabilities for text-based and single-image queries, video processing, and text-only generation. The project integrates with ComfyUI's node-based workflow system through two main node classes.

## Architecture

The extension consists of two main node classes in `nodes.py`:

- **QwenVL**: Handles vision-language model inference with support for image and video inputs
- **Qwen**: Handles text-only language model inference

Both classes follow the same architectural pattern:
- Model loading with quantization support (none, 4bit, 8bit)
- Automatic model downloading to `ComfyUI/models/LLM/`
- Device detection and optimization (CUDA/CPU with bfloat16 support)
- Memory management with optional model persistence

### Key Components

- **Model Management**: Automatic downloading from HuggingFace, local caching, quantization
- **Input Processing**: Tensor-to-PIL conversion, video preprocessing with FFmpeg
- **Inference Pipeline**: Template application, tokenization, generation, decoding
- **Memory Optimization**: Conditional model unloading and CUDA cache cleanup

## Development Commands

### Installation
```bash
# Clone and install dependencies
git clone https://github.com/alexcong/ComfyUI_QwenVL.git
cd ComfyUI_QwenVL
pip install -r requirements.txt
```

### Testing
```bash
# Run unit tests (requires PyTorch for full functionality)
python -m unittest tests.test_nodes -v

# Tests use mocking to isolate functionality from ComfyUI runtime dependencies
# Key test areas:
# - Model loading/unloading behavior
# - Memory management validation
# - Quantization configuration
# - Error handling scenarios
```

### Dependencies Management
```bash
# Install/update dependencies
pip install -r requirements.txt

# Key dependencies include:
# - torch, torchvision, numpy, pillow
# - huggingface_hub, transformers>=4.57.1, bitsandbytes, accelerate
# - qwen-vl-utils, optimum
```

## Model Configuration

### Supported Models
- **Vision-Language**: Qwen2.5-VL-3B/7B-Instruct, Qwen3-VL-2B/4B/8B/32B-Instruct, Qwen3-VL-2B/4B/8B/32B-Thinking, SkyCaptioner-V1
- **Text-Only**: Qwen2.5-3B/7B/14B/32B-Instruct, Qwen3-4B-Thinking-2507, Qwen3-4B-Instruct-2507

### Model Location
Models are automatically downloaded to: `ComfyUI/models/LLM/`

### Model ID Formats
- **Qwen3 models**: Use `Qwen/{model_name}` format (e.g., `Qwen/Qwen3-VL-4B-Thinking`)
- **Qwen2.5 models**: Use `qwen/{model_name}` format (e.g., `qwen/Qwen2.5-VL-3B-Instruct`)
- **Skywork models**: Use `Skywork/{model_name}` format (e.g., `Skywork/SkyCaptioner-V1`)

### Quantization Options
- **none**: Full precision (bfloat16/float16 based on GPU capability)
- **4bit**: BitsAndBytes 4-bit quantization
- **8bit**: BitsAndBytes 8-bit quantization

## Code Structure

### File Organization
```
├── nodes.py              # Main node implementations (QwenVL, Qwen) (~432 lines)
├── __init__.py          # Node class mappings for ComfyUI
├── pyproject.toml       # Project metadata and dependencies
├── requirements.txt     # Python dependencies
├── tests/test_nodes.py  # Unit tests with mocking framework
├── workflow/            # Example ComfyUI workflows
│   ├── Qwen2VL.json     # Multimodal workflow example
│   ├── qwen25.json      # Text generation workflow example
│   └── *.png           # Workflow screenshots
└── README.md           # User documentation
```

### Node Implementation Pattern
Both nodes follow ComfyUI's standard pattern:
- `INPUT_TYPES()`: Define input parameters and types
- `inference()`: Main processing method
- Model loading and caching in instance variables
- Device detection and optimization

## Important Implementation Details

### Model Detection and Loading Strategy
The codebase uses intelligent model detection:
```python
# For vision-language models
if model.startswith("Qwen3"):
    self.model = Qwen3VLForConditionalGeneration.from_pretrained(...)
else:
    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(...)
```

### Video Processing Pipeline
- Uses FFmpeg subprocess calls for video frame extraction and resizing
- Processes videos to 1fps with max dimension 256px for efficiency
- Creates temporary files in `/tmp/` with UUID-based unique identifiers
- Automatically cleans up temporary files after inference in finally blocks

### Memory Management Architecture
- `keep_model_loaded` parameter controls model persistence between runs
- Automatic CUDA cache cleanup when unloading models via `_clear_cuda_memory()`
- Device detection for optimal dtype selection (bfloat16 vs float16)
- ComfyUI integration: uses `comfy.model_management.soft_empty_cache()` when available

### Error Handling Strategy
- Graceful handling of empty prompts with descriptive error messages
- FFmpeg error handling for video processing failures
- Model inference exception catching with error propagation
- Resource cleanup in finally blocks to prevent memory leaks

## Development Notes

### Testing Framework
The project uses a sophisticated testing approach:
- **Mocking Strategy**: Mocks transformers, processors, and models for isolated testing
- **ComfyUI Runtime Handling**: Creates dummy `folder_paths` module when ComfyUI unavailable
- **Memory Testing**: Validates model loading/unloading behavior
- **Temporary Directories**: Uses temp directories for isolated test environments
- **Conditional Testing**: Skips tests if PyTorch not available

### Adding New Models
1. Add model name to the appropriate model list in `INPUT_TYPES()`
2. Update model_id logic if needed (Qwen vs Skywork prefixes)
3. For Qwen3 models, ensure `model.startswith("Qwen3")` detection works
4. Test model downloading and inference
5. Update documentation if new model families are added

### ComfyUI Integration
- Nodes are registered in `__init__.py` with multiple name mappings for compatibility
- Category is set to "Comfyui_QwenVL"
- Return type is always "STRING" for generated text
- Optional integration with ComfyUI's model management system

### Dependencies and Compatibility
- Requires `transformers>=4.57.1` for Qwen3VL support
- Uses `qwen-vl-utils` for vision processing
- BitsAndBytes for quantization support
- Always use the provided requirements.txt for compatibility

### Performance Considerations
- Video processing is resource-intensive; consider input size limitations
- Model loading is expensive; use `keep_model_loaded=True` for repeated inference
- Quantization significantly reduces memory usage but may affect output quality
- CUDA memory management is critical for multi-model workflows

### Image Embedding Optimization (New in v1.0)
The QwenVL node now includes significant optimizations for faster image processing:

#### New Parameters
- **`image_quality`**: Controls image resolution and processing speed
  - `fast`: ~3-4x faster, good for simple captions
  - `balanced`: ~2x faster (recommended default)
  - `high`: Good quality/speed balance
  - `ultra`: Maximum detail (original behavior)

- **`enable_image_cache`**: Caches processed embeddings (default: True)
  - Near-instant processing for repeated images
  - Automatically cleared on model unload

#### Performance Improvements
- **Dynamic Resolution**: Adjustable max_pixels based on quality setting
- **Optimized GPU Transfer**: Non-blocking CUDA operations
- **Vision Encoder Compilation**: Automatic torch.compile() on PyTorch 2.0+
- **Performance Monitoring**: Real-time timing and optimization suggestions

#### Typical Speedups
- `balanced` quality: ~2x faster than original
- `fast` quality: ~3-4x faster than original
- Repeated images with cache: ~350x faster
- Combined optimizations: 2-3x overall speedup

See `PERFORMANCE_OPTIMIZATION.md` for detailed benchmarks and usage guide.

#### Implementation Notes
- Vision encoder compilation (lines 536-546): Attempts torch.compile() for 15-30% speedup
- Image processing optimization (lines 647-695): Optimized tensor transfer and timing
- Quality presets (lines 417-423): Configurable min/max pixel settings
- Cache system (lines 275-277, 287-292): MD5-based embedding cache